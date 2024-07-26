from typing import Optional

import torch
from torch import Tensor
import torch as T
from recsys_metrics.utils import _check_ranking_inputs, _reduce_tensor, div_no_nan


def _dcg(target: Tensor) -> Tensor:
    batch_size, k = target.shape
    rank_positions = torch.arange(1, k + 1, dtype=torch.float32, device=target.device).tile((batch_size, 1))
    return (target / torch.log2(rank_positions + 1)).sum(dim=-1)


def normalized_dcg(preds: Tensor, target: Tensor, k: Optional[int] = None, reduction: Optional[str] = 'mean') -> Tensor:
    preds, target, k = _check_ranking_inputs(preds, target, k=k, batched=True)

    _, topk_idx = preds.topk(k, dim=-1)
    sorted_target = target.take_along_dim(topk_idx, dim=-1)
    ideal_target, _ = target.topk(k, dim=-1)

    return _reduce_tensor(div_no_nan(_dcg(sorted_target), _dcg(ideal_target)), reduction=reduction)

def _dcg_pi(target: Tensor) -> Tensor:
    batch_size, k = target.shape
    rank_positions = torch.arange(1, k + 1, dtype=torch.float32, device=target.device).tile((batch_size, 1))
    return (target / torch.log2(rank_positions + 1))

def _inv_dcg_pi(target: Tensor) -> Tensor:
    batch_size, k = target.shape
    rank_positions = torch.arange(1, k + 1, dtype=torch.float32, device=target.device).tile((batch_size, 1))
    return (target * torch.log2(rank_positions + 1))

def dcg_ind(preds: Tensor, target: Tensor, k: Optional[int] = None, reduction: Optional[str] = 'mean') -> Tensor:
    preds, target, k = _check_ranking_inputs(preds, target, k=k, batched=True)

    _, topk_idx = preds.topk(k, dim=-1)
    #sorted_target = target.take_along_dim(topk_idx, dim=-1)
    sorted_target = target.gather(index = topk_idx, dim=-1)
    #ideal_target, _ = target.topk(k, dim=-1)
    dcg_val = _dcg_pi(sorted_target)
    rst = T.zeros_like(preds)
    rst.scatter_(dim=1, index= topk_idx, src= dcg_val)
    #return _reduce_tensor(div_no_nan(_dcg(sorted_target), _dcg(ideal_target)), reduction=reduction)
    return rst


def wpdcg_ind(preds: Tensor, target: Tensor, k: Optional[int] = None, reduction: Optional[str] = 'mean') -> Tensor:
    preds, target, k = _check_ranking_inputs(preds, target, k=k, batched=True)

    _, topk_idx = preds.topk(k, dim=-1)
    #sorted_target = target.take_along_dim(topk_idx, dim=-1)
    sorted_target = target.gather(index = topk_idx, dim=-1)
    #ideal_target, _ = target.topk(k, dim=-1)
    dcg_val = _inv_dcg_pi(sorted_target)
    rst = T.zeros_like(preds)
    rst.scatter_(dim=1, index= topk_idx, src= dcg_val)
    #return _reduce_tensor(div_no_nan(_dcg(sorted_target), _dcg(ideal_target)), reduction=reduction)
    return rst
