import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, confusion_matrix_image
from logger.visualization import TensorboardWriter

import copy
import sys
import random
import time


def get_top_k(x, ratio):
    """it will sample the top 1-ratio of the samples."""
    x_data = x.view(-1)
    x_len = x_data.nelement()
    top_k = max(1, int(x_len * (1 - ratio)))

    # get indices and the corresponding values
    if top_k <= 1:
        if random.random() < top_k:
            _, selected_indices = torch.max(x_data.abs(), dim=0, keepdim=True)
        else:
            return None, None
    else:
        _, selected_indices = torch.topk(
            x_data.abs(), int(top_k), largest=True, sorted=False
        )
    return x_data[selected_indices], selected_indices


def get_mask(flatten_arr, indices):
    mask = torch.zeros_like(flatten_arr)
    mask[indices] = 1
    mask = mask.bool()
    return mask.float(), (~mask).float()


class ProactiveTrainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.validation = 'prequential'
        self.log_step = int(np.sqrt(100))
        self.deployed_model = copy.deepcopy(self.model)


        self.deployed_model.eval()

        log_dir = str(config.log_dir)
        self.train_writer = TensorboardWriter(log_dir + '/train', self.logger, config['trainer'])
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.train_writer)
        self.train_metrics_cls = [m() for m in self.metric_ftns]

        self.init_model = copy.deepcopy(self.model)
        self.init_model.eval()
        self.init_writer = TensorboardWriter(log_dir + '/prequential_init', self.logger, config['trainer'])
        self.init_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.init_writer)
        self.init_metrics_cls = [m() for m in self.metric_ftns]

        self.deployed_writer = TensorboardWriter(log_dir + '/prequential_deployed', self.logger, config['trainer'])
        self.deployed_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns],
                                              writer=self.deployed_writer)
        self.deployed_metrics_cls = [m() for m in self.metric_ftns]

        self.trained_writer = TensorboardWriter(log_dir + '/prequential_trained', self.logger, config['trainer'])
        self.trained_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns],
                                             writer=self.trained_writer)
        self.trained_metrics_cls = [m() for m in self.metric_ftns]

        self.models_and_metrics = {
            'init': (self.init_model, self.init_metrics, self.init_metrics_cls),
            'deployed': (self.deployed_model, self.deployed_metrics, self.deployed_metrics_cls),
            'trained': (self.model, self.trained_metrics, self.trained_metrics_cls)}
        self.batch_idx = 0
        self.prequential_log = {}

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.train_metrics.reset()
        total_time = 0
        for self.batch_idx, (batch, new_batch) in enumerate(
                zip(self.data_loader, self.data_loader.streaming_dataloader)):
            (data, target) = batch
            data, target = data.to(self.device), target.to(self.device)
            if self.validation == 'prequential':
                self._prequential_evaluation(new_batch)
            self.model.train()

            start_time = time.time()

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.train_metrics.writer.set_step(self.batch_idx, 'train')
            self.train_metrics.update('loss', loss.item())
            for met in self.train_metrics_cls:
                met.update(output, target)
                self.train_metrics.update(met.__class__.__name__, met.result())

            ratio = 1 - self.config['deployment_ratio']
            ps = []
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if hasattr(p, 'residue'):
                        p.grad += p.residue
                        ps.append(p)
                    else:
                        ps.append(p)
            total = 0
            for (p, param, dparam) in zip(ps, self.model.parameters(), self.deployed_model.parameters()):
                if param.requires_grad:
                    orig_shape = p.grad.shape
                    # func is either get_random_k, get_top_k
                    data, ind = get_top_k(p.grad, ratio)
                    if ind is None:
                        print("No parameteres changed. Continuing...")
                        continue
                    total += ind.nelement()
                    mask, inv_mask = get_mask(p.grad.view(-1), ind)
                    p.mask = mask
                    p.inv_mask = inv_mask
                    p.residue = torch.reshape((p.grad.view(-1) * p.inv_mask), orig_shape)
                    dparam.view(-1)[ind] = param.view(-1)[ind]
                    # if not hasattr(param, 'start'):
                    #     param.start = 0
                    # l = param.nelement()
                    # param.end = min(param.start + max(1, int(l * ratio)), l - 1)
                    # # print(l, ratio, param.start, param.end)
                    # # print(param.view(-1)[param.start:param.end])
                    # param.start = (param.end + 1) % (l - 1)
                    # dparam.view(-1)[param.start:param.end] = param.view(-1)[param.start:param.end]
            if self.batch_idx == 10:
                print("Total parameters changed", total)
            # for (param, dparam) in zip(self.model.parameters(), self.deployed_model.parameters()):
            #     if param.requires_grad:
            #         if not hasattr(param, 'start'):
            #             param.start = 0
            #         l = param.nelement()
            #         param.end = min(param.start + max(1, int(l * ratio)), l - 1)
            #         # print(l, ratio, param.start, param.end)
            #         # print(param.view(-1)[param.start:param.end])
            #         param.start = (param.end + 1) % (l - 1)
            #         dparam.view(-1)[param.start:param.end] = param.view(-1)[param.start:param.end]

            total_time += time.time() - start_time

            if self.batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} {} Loss: {:.6f} Time per batch (ms) {}'.format(
                    epoch,
                    self._progress(self.batch_idx),
                    loss.item(), total_time * 1000 / (self.batch_idx + 1)),)
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                # self.writer.add_figure('confusion_matrix', confusion_matrix_image(output, target))
                # valid_log = self._valid_deployed(batch_idx)
                # print logged informations to the screen
                # for key, value in valid_log.items():
                #     self.logger.info('Valid deployed    {:15s}: {}'.format(str(key), value))

        log = self.train_metrics.result()

        # if self.validation == 'prequential':
        #     for key, (model, metrics, metrics_cls) in self.models_and_metrics.items():
        #         print(key, metrics_cls.result())
        # log.update(**{(key + 'val' + k): v for k, v in metrics.result()})

        return log

    def _prequential_evaluation(self, batch):
        """
        Validate  training an epoch

        :param batch: tuple (data, target) containing current training batch.
        """
        with torch.no_grad():
            for key, (model, metrics, metrics_cls) in self.models_and_metrics.items():
                model.eval()
                data, target = batch
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = self.criterion(output, target)
                metrics.writer.set_step(self.batch_idx)
                metrics.update('loss', loss.item())
                for met in metrics_cls:
                    met.update(output, target)
                    metrics.update(met.__class__.__name__, met.result())
            # self.prequential_log.update()
        # with torch.no_grad():
        #     for batch_idx, (data, target) in enumerate(self.valid_data_loader):

        # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
        # self.writer.add_figure('confusion_matrix', confusion_matrix_image(output, target))

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        # return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = len(self.data_loader)
        return base.format(current, total, 100.0 * current / total)
