from turtle import forward
import torch as T
import torch.nn as  nn
import numpy as np

class pop_rerank(nn.Module):
    def __init__(self, learner, counts, thresholdL, thresholdH, ): #x_items
        super().__init__()
        self.learner = learner

        """if x_items.is_sparse:
            self.counts = (T.sparse.sum(x_items, dim=1).to_dense())
        else:
            self.counts = (T.sum(x_items, dim=1))"""
        self.counts = counts
        self.thresholdL = thresholdL
        self.thresholdH = thresholdH

        _, self.indices = T.sort(self.counts, dim=0, descending=False)
        
        self.low_rank = 1-self.indices/ T.max(self.indices)

    def forward(self, X):
        result = self.learner(X)

        cond = T.gt(result, self.thresholdH)
        
        proc_result = T.where(cond, result + 1e1, self.low_rank)

        cond = T.gt(result, self.thresholdL)

        proc_result = T.where(cond, proc_result, T.zeros_like(proc_result))
        return proc_result

class rand_rerank(nn.Module):
    def __init__(self, learner, x_items, threshold):
        super().__init__()
        self.learner = learner

        if x_items.is_sparse:
            self.counts = (T.sparse.sum(x_items, dim=1).to_dense())
        else:
            self.counts = (T.sum(x_items, dim=1))
        self.threshold = threshold
        _, self.indices = T.sort(self.counts, dim=0, descending=False)
        
        self.low_rank = 1-self.indices/ self.counts.shape[0]

    def forward(self, X):
        result = self.learner(X)

        cond = T.gt(result, self.threshold)
        
        proc_result = T.where(cond, result + 1e1, T.rand(X.shape,))
        return proc_result