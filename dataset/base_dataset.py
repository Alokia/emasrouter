from abc import ABC, abstractmethod
from typing import List, Union
import random
from itertools import accumulate
import math


class BaseDataset(ABC):
    def __init__(self, data: List = None):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def split(data: List, ratio: Union[List, float], shuffle: bool = True, **kwargs) -> List[List]:
        if shuffle:
            random.shuffle(data)
        if isinstance(ratio, float):
            ratio = [ratio, 1 - ratio]
        ratio = list(accumulate(ratio))
        split_idx = [0] + [round(len(data) * r) for r in ratio]

        split_data = []
        for i in range(len(split_idx) - 1):
            split_data.append(data[split_idx[i]:split_idx[i + 1]])
        assert sum([len(i) for i in split_data]) == len(data)
        return split_data


class BaseDataLoader(ABC):
    def __init__(self, dataset: BaseDataset, batch_size: int = 1, shuffle: bool = True, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.n_iter = 0
        self.indices = list(range(len(self.dataset)))
        # Pre-emptive disruption to avoid using `next` directly to produce sequential results
        if self.shuffle:
            random.shuffle(self.indices)

    def __len__(self):
        if self.drop_last:
            return math.floor(len(self.dataset) / self.batch_size)
        else:
            return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        self.n_iter = 0
        for i in range(len(self)):
            start_index, end_index = i * self.batch_size, (i + 1) * self.batch_size
            batch = [self.dataset[j] for j in self.indices[start_index:end_index]]
            self.n_iter += 1
            yield batch

    def __next__(self):
        if self.n_iter >= len(self):
            raise StopIteration

        start_index, end_index = self.n_iter * self.batch_size, (self.n_iter + 1) * self.batch_size
        batch = [self.dataset[j] for j in self.indices[start_index:end_index]]
        self.n_iter += 1
        return batch


class BaseMetric(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass
