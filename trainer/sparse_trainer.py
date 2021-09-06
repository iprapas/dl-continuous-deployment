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
import math


def get_top_k(x, ratio):
    """it will sample the top 1-ratio of the samples."""
    x_data = x.view(-1)
    x_len = x_data.nelement()
    top_k = max(1, x_len * (1 - ratio))

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


def get_random_k(x, ratio):
    """it will randomly sample the 1-ratio of the samples."""
    # get tensor size.
    x_data = x.view(-1)
    x_len = x_data.nelement()
    top_k = max(1, int(x_len * (1 - ratio)))

    # random sample the k indices.
    selected_indices = np.random.choice(x_len, top_k, replace=False)
    selected_indices = torch.LongTensor(selected_indices).to(x.device)

    return x_data[selected_indices], selected_indices


def get_n_bits(tensor, quantize_level=32):
    return 8 * tensor.nelement() * tensor.element_size() * quantize_level / 32


def get_qsgd(x, q_level, is_biased=False):
    norm = x.norm(p=2)
    s = 2 ** q_level - 1
    level_float = x.abs() / norm * s
    previous_level = torch.floor(level_float)
    is_next_level = (torch.rand_like(x) < (level_float - previous_level)).float()
    new_level = previous_level + is_next_level

    scale = 1
    if is_biased:
        d = x.nelement()
        scale = 1.0 / (min(d / (s ** 2), math.sqrt(d) / s) + 1.0)
    return scale / s * torch.sign(x) * norm * new_level


class SparseTrainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader, valid_data_loader):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.validation = 'prequential'
        self.log_step = 10
        self.valid_data_loader = valid_data_loader
        self.history_end = self.data_loader.history_end
        log_dir = str(config.log_dir)
        self.train_writer = TensorboardWriter(log_dir + '/train', self.logger, config['trainer'])
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], 'time_per_batch',
                                           'parameters_changed', writer=self.train_writer)
        self.train_metrics_cls = [m() for m in self.metric_ftns]

        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics_cls = [m() for m in self.metric_ftns]

        self.trained_writer = TensorboardWriter(log_dir + '/prequential_trained', self.logger, config['trainer'])
        self.trained_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns],
                                             writer=self.trained_writer)
        self.trained_metrics_cls = [m() for m in self.metric_ftns]

        self.models_and_metrics = {
            'trained': (self.model, self.trained_metrics, self.trained_metrics_cls),
        }

        if self.config['init_tracking']:
            self.logger.info("Tracking initial (not being trained) model")
            self.init_model = copy.deepcopy(self.model)
            self.init_model.eval()
            self.init_writer = TensorboardWriter(log_dir + '/prequential_init', self.logger, config['trainer'])
            self.init_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.init_writer)
            self.init_metrics_cls = [m() for m in self.metric_ftns]
            self.models_and_metrics['init'] = (self.init_model, self.init_metrics, self.init_metrics_cls)

        self.batch_idx = 0
        self.ratio = 1 - self.config['deployment_ratio']
        self.is_online = self.config['online_training']
        self.trigger_size = self.config['data_loader']['args']['trigger_size']

        self.prequential_log = {}
        # compression type for sparse changes: 'no', 'cycle' or 'topk'
        self.compress_type = self.config['compress_type']
        # self.compress_memory = self.config['compress_memory']
        if self.compress_type == 'topk':
            self.compress_function = lambda x: get_top_k(x, self.ratio)
        elif self.compress_type == 'randomk':
            self.compress_function = lambda x: get_random_k(x, self.ratio)
        elif self.compress_type.startswith('quant'):
            self.quant_level = int(self.compress_type.split('_')[-1])
            assert self.quant_level in [2 ** x for x in range(5)]
            self.compress_function = lambda x: get_qsgd(x, q_level=self.quant_level, is_biased=False)

        self.memory_of_grads = dict()

        for i, group in enumerate(self.optimizer.param_groups):
            for j, p in enumerate(group["params"]):
                if p.requires_grad:
                    self.memory_of_grads[(i, j)] = torch.zeros_like(p.data)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.train_metrics.reset()
        total_time = 0
        start_time = time.time()

        for self.batch_idx, (batch, new_batch) in enumerate(
                zip(self.data_loader, self.data_loader.streaming_dataloader)):
            if self.is_online:
                (data, target) = new_batch
            else:
                (data, target) = batch
            data, target = data.to(self.device), target.to(self.device)
            if self.validation == 'prequential':
                self._prequential_evaluation(new_batch)

            self.model.train()
            batch_start = time.time()

            if self.compress_type == 'no':
                self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.criterion(output, target)
            # calculate gradients
            loss.backward()
            total_parameters = 0
            total_changed = 0
            total_bits = 0
            # unmask top k gradients
            if self.compress_type in ['topk', 'randomk']:
                for group in self.optimizer.param_groups:
                    for p in group["params"]:
                        if p.requires_grad:
                            data, ind = self.compress_function(p.grad)
                            if ind is None:
                                print("No parameteres changed. Continuing...")
                                continue

                            # sparsify gradients and keep residues
                            p.residue = p.grad.clone()
                            p.residue.view(-1)[ind] = 0
                            p.grad = p.grad - p.residue

                            # total_bits += get_n_bits(data) + get_n_bits(ind)
                            #
                            # total_parameters += p.grad.nelement()
                            # total_changed += (p.grad != 0).sum()
            elif self.compress_type.startswith('quant'):
                for i, group in enumerate(self.optimizer.param_groups):
                    for j, p in enumerate(group["params"]):
                        if p.requires_grad:
                            q_grad = self.compress_function(p.grad)
                            # sparsify gradients and keep residues
                            p.residue = p.grad - q_grad
                            p.grad = q_grad
                            # total_bits += get_n_bits(q_grad, self.quant_level)
                            # total_parameters += p.grad.nelement()
                            # total_changed += (p.grad != 0).sum()
            # cyclek
            # dparam.view(-1)[p.start:p.end] = p.view(-1)[p.start:p.end]

            # do gradient descent
            self.optimizer.step()

            batch_time = time.time() - batch_start
            total_time += time.time() - batch_start

            # unmask bottom gradients
            for p in self.model.parameters():
                if p.requires_grad:
                    total_bits += get_n_bits(p.grad)
                    total_changed += (p.grad != 0).sum()
                    total_parameters += p.grad.nelement()
                    if not (self.compress_type == 'no'):
                        p.grad = p.residue

            step = (self.batch_idx + 1) * self.trigger_size + self.history_end
            self.train_metrics.writer.set_step(step, 'train')
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('time_per_batch', batch_time)
            self.train_metrics.update('parameters_changed', total_changed)
            for met in self.train_metrics_cls:
                met.update(output, target)
                self.train_metrics.update(met.__class__.__name__, met.result())

            if self.batch_idx % self.log_step == 0:
                self.logger.info(
                    "Percentage of parameters changed: {:.2f} %".format(
                        100 * int(total_changed) / int(total_parameters + 1)))
                self.logger.info('Train Epoch: {} {} Loss: {:.6f} Time per batch (ms): {}'.format(
                    epoch,
                    self._progress(self.batch_idx),
                    loss.item(), total_time * 1000 / (self.batch_idx + 1)))

        log = self.train_metrics.result()
        log['time'] = time.time() - start_time
        log['training_time'] = total_time
        if self.valid_data_loader:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
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
                step = self.batch_idx * self.trigger_size + self.history_end + len(target)
                metrics.writer.set_step(step, "test")
                metrics.update('loss', loss.item())
                for met in metrics_cls:
                    met.update(output, target)
                    metrics.update(met.__class__.__name__, met.result())

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = len(self.data_loader)
        return base.format(current, total, 100.0 * current / total)

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step(len(target) * batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.valid_metrics_cls:
                    met.update(output, target)
                    self.valid_metrics.update(met.__class__.__name__, met.result())

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()
