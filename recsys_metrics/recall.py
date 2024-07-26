from typing import Optional

from torch import Tensor

import torch as T
from recsys_metrics.utils import _check_ranking_inputs, _reduce_tensor, div_no_nan


def recall(preds: Tensor, target: Tensor, k: Optional[int] = None, reduction: Optional[str] = 'mean') -> Tensor:
    preds, target, k = _check_ranking_inputs(preds, target, k=k, batched=True)
    _, topk_idx = preds.topk(k, dim=-1)
    relevance = target.take_along_dim(topk_idx, dim=-1).sum(dim=-1).float()
    assert k is not None
    tsm  = T.clamp(target.sum(dim=-1), max=k)
    return _reduce_tensor(div_no_nan(relevance, tsm), reduction=reduction)

def recall_ind(preds: Tensor, target: Tensor, k: Optional[int] = None, reduction: Optional[str] = 'none') -> Tensor:
    preds, target, k = _check_ranking_inputs(preds, target, k=k, batched=True)
    _, topk_idx = preds.topk(k, dim=-1)
    rst = T.zeros_like(preds)
    rst.scatter_(dim=1, index= topk_idx, src= T.gather(target.float(), dim=1, index=topk_idx))
    #return _reduce_tensor(div_no_nan(relevance, target.sum(dim=-1)), reduction=reduction)
    return rst

def freq_ind(preds: Tensor, target: Tensor, k: Optional[int] = None, reduction: Optional[str] = 'none') -> Tensor:
    preds, target, k = _check_ranking_inputs(preds, target, k=k, batched=True)
    _, topk_idx = preds.topk(k, dim=-1)
    rst = T.zeros_like(preds)
    rst.scatter_(dim=1, index= topk_idx, src= T.ones_like(topk_idx).float())
    #return _reduce_tensor(div_no_nan(relevance, target.sum(dim=-1)), reduction=reduction)
    return rst

def recall_ind_dist(preds: Tensor, target: Tensor, k: Optional[int] = None, reduction: Optional[str] = 'none') -> Tensor:
    preds, target, k = _check_ranking_inputs(preds, target, k=k, batched=True)
    _, topk_idx = preds.topk(k, dim=-1)
    rst = T.zeros_like(preds)
    relevance = target.take_along_dim(topk_idx, dim=-1).float()#.sum(dim=-1,keepdim=True)
    relevance = relevance + 1e-8
    tsm  = T.clamp(target.sum(dim=-1, keepdim=True), max=k) + 1e-8
    #rst.scatter_(dim=1, index= topk_idx, src= T.gather(relevance/ tsm, dim=1, index=topk_idx))
    rst.scatter_(dim=1, index= topk_idx, src= relevance/ tsm)
    #return _reduce_tensor(div_no_nan(relevance, target.sum(dim=-1)), reduction=reduction)
    return rst