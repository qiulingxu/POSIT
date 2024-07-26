
from abc import ABC, abstractmethod
from collections.abc import Iterable
import numpy as np

class Metric(ABC):

    def calc_batch_profile(self, pred, profile, **karg):
        return self.calc_batch(pred, label = profile["label"].to_numpy(), weight = profile["weight"].to_numpy(), itemid = profile["show_title_id"].to_numpy(), **karg)

    @abstractmethod
    def save_bucket_type(self):
        assert False
        return list
    
    def save_bucket_instance(self):
        return (self.save_bucket_type())()

    @abstractmethod
    def calc_batch(self, pred, **karg):
        assert False
    
    
    def aggregate(self, xs):
        #if isinstance(xs, list):
        #    if isinstance(xs[0], Iterable):
        #        xs = np.concatenate(xs, axis=0)
        return self._aggregate(xs)

    @abstractmethod
    def _aggregate(self, xs):
        assert False

    @abstractmethod
    def add(self, save_bucket, x):
        assert False

