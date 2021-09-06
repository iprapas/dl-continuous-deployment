from torch.utils.data import IterableDataset, Sampler, DataLoader
import torch
import numpy as np


class BatchRandomDynamicSampler(Sampler):
    r"""Implements proactive training window-sampling
    :attr:`data_source`
    :attr:`batch_size`
    :attr:`history_idx`
    :attr:`trigger_size
    :attr:`replacement`
    """

    def __init__(self, data_source, history_idx=0, batch_size=1, trigger_size=1, replacement=False):
        super().__init__(data_source)
        self.data_source = data_source
        self.replacement = replacement
        self.history_idx = history_idx
        self.batch_size = batch_size
        self.trigger_size = trigger_size

        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

    @property
    def num_samples(self):
        # dataset size will change at runtime
        return self.history_idx

    def __iter__(self):
        n = len(self.data_source)
        while self.history_idx < n:
            self.history_idx += self.trigger_size
            # for i in range(self.batch_size):
            #     batch.append(torch.randint(high=self.history_idx, size=(1,)))
            if self.replacement:
                yield (np.random.choice(range(self.history_idx), size=self.batch_size, replace=False, p=None))
            else:
                yield (torch.randint(high=self.history_idx, size=(self.batch_size,)))

        #     yield
        # if self.replacement:
        #     return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        # return iter(torch.randperm(n).tolist())

    def __len__(self):
        return self.num_samples

    def get_new_batch(self):
        print(self.history_idx, self.trigger_size)
        start = self.history_idx - self.trigger_size + 1
        end  = self.history_idx + 1
        end = min(len(self.data_source), end)
        return list(range(start, end, 1))


if __name__ == '__main__':
    data = list(range(10, 20))
    sampler = BatchRandomDynamicSampler(data, 4, 4, 3, False)


    dataloader = torch.utils.data.DataLoader(data, batch_sampler=sampler)
    for i, item in enumerate(dataloader):
        print("new batch", sampler.get_new_batch())
        print(len(dataloader))
        print(i, item)
