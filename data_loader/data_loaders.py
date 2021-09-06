from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import ConcatDataset, IterableDataset, Sampler, DataLoader
import torch
import numpy as np

from PIL import Image
import os
import os.path
import numpy as np
import pickle
from torchvision import datasets
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


def softscale(x):
    a = 1 / x
    return a / a.sum()


class BatchRandomDynamicSampler(Sampler):
    r"""Implements proactive training window-sampling
    :attr:`data_source`
    :attr:`batch_size`
    :attr:`history_idx`
    :attr:`trigger_size
    :attr:`replacement`
    """

    def __init__(self, data_source, history_idx=0, batch_size=1, trigger_size=1, window_size=0, replacement=True,
                 always_include_new=True, shuffle=False, keep_probs=True):
        super().__init__(data_source)
        self.data_source = data_source
        self.replacement = replacement
        self.history_idx = history_idx
        self.batch_size = batch_size
        self.trigger_size = trigger_size
        self.window_size = max(window_size, batch_size) if window_size else 0
        self.always_include_new = always_include_new
        self.shuffle = shuffle
        self.keep_probs = keep_probs
        if self.keep_probs:
            self.times_selected = np.ones(self.history_idx) * 50

        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

    @property
    def num_samples(self):
        # dataset size will change at runtime
        return self.history_idx

    def _get_probs(self):
        if self.keep_probs:
            return softscale(self.times_selected)
        else:
            return None

    def __iter__(self):
        n = len(self.data_source)
        while self.history_idx < n:
            sample_history_idx = self.history_idx

            self.history_idx += self.trigger_size
            self.history_idx = min(self.history_idx, n)
            # for i in range(self.batch_size):
            #     batch.append(torch.randint(high=self.history_idx, size=(1,)))
            if self.replacement:
                if self.always_include_new:
                    if self.keep_probs:
                        self.times_selected = np.append(self.times_selected, np.ones(self.trigger_size))
                    new_elements = np.arange(sample_history_idx, self.history_idx, 1)
                    start_sample = 0
                    if self.window_size and (sample_history_idx > self.window_size):
                        start_sample = sample_history_idx - self.window_size
                    end_sample = sample_history_idx
                    old_elements = np.random.choice(np.arange(start_sample, end_sample, 1)
                                                    , size=self.batch_size - self.trigger_size, replace=False,
                                                    p=self._get_probs())
                    result = np.concatenate((new_elements, old_elements))
                    if self.keep_probs:
                        self.times_selected[old_elements] += 1
                else:
                    if self.keep_probs:
                        self.times_selected = np.append(self.times_selected, np.ones(self.trigger_size) / 1000)
                    result = np.random.choice(np.arange(self.history_idx)
                                              , size=self.batch_size, replace=False, p=self._get_probs())
                    if self.keep_probs:
                        self.times_selected[result] += 1
            else:
                result = torch.randint(high=self.history_idx, size=(self.batch_size,))
            if self.shuffle:
                np.random.shuffle(result)
                yield result
            else:
                yield result

        #     yield
        # if self.replacement:
        #     return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        # return iter(torch.randperm(n).tolist())

    def __len__(self):
        return self.num_samples


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class CifarDataLoader(BaseDataLoader):
    """
    CIFAR data loading using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, start=0, end=10000, num_workers=1,
                 training=True):
        trsfm = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])

        self.data_dir = data_dir
        self.start = start
        self.end = end
        self.dataset = CIFAR10_Init(self.data_dir, train=training, start=start, end=end, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class CIFAR10_Init(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train ('train', 'initial', 'test'): If 'train', creates dataset from training set,
        if 'initial' creates a set from the first batch, otherwise
            creates from 'test' set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key'     : 'label_names',
        'md5'     : '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train=True, transform=None, target_transform=None, start=0, end=0,
                 download=False):

        super(CIFAR10_Init, self).__init__(root, transform=transform,
                                           target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        if start or end:
            assert (start >= 0), "start index should be greater than "
            assert (end > start), "end index should be larger than start index"
            assert (end <= len(self.data)), "end index should not be larger than size of dataset"
            self.data = self.data[start:end]
            self.targets = self.targets[start:end]
        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


CIFAR10_TRAIN_SIZE = 50000


class CifarDataLoaderProactive(DataLoader):
    """
    CIFAR data loading using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, trigger_size=10, window_size=0, num_workers=1, always_include_new=False,
                 keep_probs=False, history_start=0, history_end=10000, stream_end=CIFAR10_TRAIN_SIZE):
        trsfm = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
        self.history_start = history_start
        self.history_end = history_end
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.historical_dataset = CIFAR10_Init(self.data_dir, train=True, start=history_start, end=history_end,
                                               download=True, transform=trsfm)
        self.streaming_dataset = CIFAR10_Init(self.data_dir, train=True, start=history_end, end=stream_end,
                                              download=True, transform=trsfm)
        self.dataset = ConcatDataset([self.historical_dataset, self.streaming_dataset])

        self.sampler = BatchRandomDynamicSampler(self.dataset, batch_size=batch_size,
                                                 history_idx=history_end - history_start,
                                                 trigger_size=trigger_size,
                                                 window_size=window_size,
                                                 always_include_new=always_include_new,
                                                 keep_probs=keep_probs)
        self.streaming_dataloader = DataLoader(self.streaming_dataset, trigger_size, False)
        super().__init__(self.dataset, batch_sampler=self.sampler, num_workers=num_workers)


class MNISTDataLoader(BaseDataLoader):
    """
    MNIST data loading using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, start=0, end=0, num_workers=1,
                 training=True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.start = start
        self.end = end
        self.data_dir = data_dir
        self.dataset = get_MNIST(self.data_dir, train=training, download=True, start=start, end=end, transform=transform)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

def get_MNIST(data_dir, train=True, start=0, end=0, download=True, transform=None):
    dataset = datasets.MNIST(data_dir, train=train,
                                                 download=True, transform=transform)
    if start or end:
        assert (start >= 0), "start index should be greater than "
        assert (end > start), "end index should be larger than start index"
        assert (end <= len(dataset.data)), "end index should not be larger than size of dataset"
        dataset.data = dataset.data[start:end]
        dataset.targets = dataset.targets[start:end]
    return dataset

MNIST_TRAIN_SIZE=60000
class MNISTDataLoaderProactive(DataLoader):
    """
    MNIST data loading using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, trigger_size=10, window_size=0, num_workers=1, always_include_new=False,
                 keep_probs=False, history_start=0, history_end=10000, stream_end=MNIST_TRAIN_SIZE):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.history_start = history_start
        self.history_end = history_end
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.historical_dataset = get_MNIST(self.data_dir, train=True, start=history_start, end=history_end,
                                                 download=True, transform=transform)

        self.streaming_dataset = get_MNIST(self.data_dir, train=True,download=True,start=history_end, end=stream_end, transform=transform)

        self.dataset = ConcatDataset([self.historical_dataset, self.streaming_dataset])

        self.sampler = BatchRandomDynamicSampler(self.dataset, batch_size=batch_size,
                                                 history_idx=history_end - history_start,
                                                 trigger_size=trigger_size,
                                                 window_size=window_size,
                                                 always_include_new=always_include_new,
                                                 keep_probs=keep_probs)
        self.streaming_dataloader = DataLoader(self.streaming_dataset, trigger_size, False)
        super().__init__(self.dataset, batch_sampler=self.sampler, num_workers=num_workers)
