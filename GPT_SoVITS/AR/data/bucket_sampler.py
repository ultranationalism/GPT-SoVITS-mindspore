# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/data/bucket_sampler.py
# reference: https://github.com/lifeiteng/vall-e
import itertools
import math
import random
from random import shuffle
from typing import Iterator
from typing import Optional
from typing import TypeVar

import mindspore as ms
from mindspore.dataset import Dataset
from mindspore.dataset import Sampler

__all__ = [
    "BucketSampler",
]


class BucketSampler(Sampler):
    r"""
    这原本是一个分布式采样器
    sort the dataset wrt. input length
    divide samples into buckets
    sort within buckets
    divide buckets into batches
    sort batches
    """

    def __init__(
        self,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        batch_size: int = 32,
    ) -> None:
        self.epoch = 0
        self.drop_last = drop_last
        self.total_size = self.num_samples
        self.shuffle = shuffle
        self.seed = seed
        self.batch_size = batch_size
        self.id_buckets = self.make_buckets(bucket_width=2.0)

    def make_buckets(self, bucket_width: float = 2.0):
        buckets = []
        cur = []
        max_sec = bucket_width
        for id, sec in self.id_with_length:
            if sec < max_sec:
                cur.append(id)
            else:
                buckets.append(cur)
                cur = [id]
                max_sec += bucket_width
        if len(cur) > 0:
            buckets.append(cur)
        return buckets

    def __iter__(self) -> Iterator:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            ms.set_seed(self.seed + self.epoch)
            random.seed(self.epoch + self.seed)
            shuffled_bucket = []
            for buc in self.id_buckets:
                buc_copy = buc.copy()
                shuffle(buc_copy)
                shuffled_bucket.append(buc_copy)
            grouped_batch_size = self.batch_size * self.num_replicas
            shuffled_bucket = list(itertools.chain(*shuffled_bucket))
            n_batch = int(math.ceil(len(shuffled_bucket) / grouped_batch_size))
            batches = [
                shuffled_bucket[b * grouped_batch_size : (b + 1) * grouped_batch_size]
                for b in range(n_batch)
            ]
            shuffle(batches)
            indices = list(itertools.chain(*batches))
        else:
            # type: ignore[arg-type]
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
