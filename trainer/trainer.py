import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, confusion_matrix_image
import copy
import sys
import time
from model.metric import Accuracy, TopkAccuracy


def get_top_k(x, ratio):
    """it will sample the top 1-ratio of the samples."""
    x_data = x.view(-1)
    x_len = x_data.nelement()
    top_k = max(1, int(x_len * (1 - ratio)))

    # get indices and the corresponding values
    if top_k == 1:
        _, selected_indices = torch.max(x_data.abs(), dim=0, keepdim=True)
    else:
        _, selected_indices = torch.topk(
            x_data.abs(), top_k, largest=True, sorted=False
        )
    return x_data[selected_indices], selected_indices


def get_mask(flatten_arr, indices):
    mask = torch.zeros_like(flatten_arr)
    mask[indices] = 1
    mask = mask.bool()
    return mask.float(), (~mask).float()


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.deployed_model = copy.deepcopy(self.model)
        self.init_model = copy.deepcopy(self.model)
        self.init_model.eval()
        self.deployed_model.eval()

        self.accuracy = Accuracy()
        self.topkaccuracy = TopkAccuracy()

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        start = time.time()
        self.model.train()
        total_batch = 0
        self.train_metrics.reset()
        training_time = 0
        for batch_idx, (data, target) in enumerate(self.data_loader):

            data, target = data.to(self.device), target.to(self.device)
            batch_start = time.time()

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            training_time += time.time() - batch_start


            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            total_batch += time.time() - batch_start
            if batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} {} Loss: {:.6f} Time per batch (ms) {}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(), total_batch * 1000 / (batch_idx + 1)))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                # self.writer.add_figure('confusion_matrix', confusion_matrix_image(output, target))
                # valid_log = self._valid_deployed(batch_idx)
                # print logged informations to the screen
                # for key, value in valid_log.items():
                #     self.logger.info('Valid deployed    {:15s}: {}'.format(str(key), value))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        log['time (sec)'] = time.time() - start
        log['training_time'] = training_time
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()

        avg_loss =0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)

                loss = self.criterion(output, target)
                avg_loss += loss.item()/len(self.valid_data_loader)

                pred = torch.argmax(output, dim=1)
                correct += torch.sum(pred == target).item()
                total += len(target)

        self.writer.set_step(epoch, 'valid')
        self.writer.add_scalar('loss', avg_loss)
        self.writer.add_scalar('accuracy', correct/total)
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                # self.writer.add_figure('confusion_matrix', confusion_matrix_image(output, target))

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _valid_deployed(self, batch):
        """
        Validate after training a batch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.deployed_model.eval()

        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((batch - 1) * len(self.valid_data_loader) + batch_idx*len(target), 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met)
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                # self.writer.add_figure('confusion_matrix', confusion_matrix_image(output, target))

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
