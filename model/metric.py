import torch
from abc import abstractmethod


class Metric(object):
    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def update(self, output, target):
        """
        Update method for metric
        :param output: Model output
        :param target: Target output
        """
        raise NotImplementedError

    @abstractmethod
    def result(self):
        """
        Returns result of metric
        """
        raise NotImplementedError


import json
import os


class Accuracy(Metric):
    def __init__(self):
        self.num_correct = 0
        self.num_examples = 0

    def reset(self):
        self.num_examples = 0
        self.num_correct = 0

    def update(self, output, target):
        with torch.no_grad():
            pred = torch.argmax(output, dim=1)
            assert pred.shape[0] == len(target)
            self.num_correct += torch.sum(pred == target).item()
            self.num_examples += len(target)

    def save(self, log_dir):
        log_file = os.path.join(log_dir, self.__class__.__name__ + '.json')
        with open(log_file, 'w') as outfile:
            json.dump(self.__dict__, outfile)

    def load(self, log_dir):
        log_file = os.path.join(log_dir, self.__class__.__name__ + '.json')
        if not os.path.isfile(log_file):
            print("File {} does not exist.".format(log_file))
            return
        with open(log_file, 'r') as infile:
            data = json.load(infile)
        self.num_correct = data['num_correct']
        self.num_examples = data['num_examples']

    def result(self):
        return self.num_correct / self.num_examples


class ExponentialMovingAverage(Metric):
    def __init__(self, func, alpha=0.9):
        assert (0 < alpha < 1)
        self.func = func
        self.res = None

    def reset(self):
        self.res = None

    def update(self, output, target):
        if self.res is None:
            self.res = self.func(output, target)
        else:
            self.res = self.alpha * self.func(output, target) + (1 - self.alpha) * self.res

    def result(self):
        return self.res


class ExponentialMovingAccuracy(ExponentialMovingAverage):
    def __init__(self, alpha=0.9):
        super().__init__(accuracy, alpha)


class TopkAccuracy(Metric):
    def __init__(self, k=3):
        self.num_correct = 0
        self.num_examples = 0
        self.k = k

    def reset(self):
        self.num_examples = 0
        self.num_correct = 0

    def update(self, output, target):
        with torch.no_grad():
            pred = torch.topk(output, self.k, dim=1)[1]
            assert pred.shape[0] == len(target)
            for i in range(self.k):
                self.num_correct += torch.sum(pred[:, i] == target).item()
            self.num_examples += len(target)

    def result(self):
        return self.num_correct / self.num_examples


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def correct(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct


def total(output, target):
    return len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
