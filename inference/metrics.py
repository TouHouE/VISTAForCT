from typing import Optional, List
import logging

import torch
import numpy as np
from monai.data import MetaTensor




@torch.no_grad()
def my_dice(y_true: torch.Tensor | MetaTensor, y_pred: torch.Tensor | MetaTensor, nc: Optional[int] = None, ignore_bg=False) -> list[float]:
    """
    Dice coefficient between two
    Args:
        y_true: Should be a tensor with shape one_hot_channel x H x W or 1 x H x W but category-digit in tensor
        y_pred: Should be a tensor with shape one_hot_channel x H x W or 1 x H x W but category-digit in tensor
        nc: number of categories is optional.
    """
    if isinstance(y_true, MetaTensor):
        y_true: torch.Tensor = y_true.as_tensor()

    if isinstance(y_pred, MetaTensor):
        y_pred: torch.Tensor = y_pred.as_tensor()
    y_true = y_true.cpu().long()
    y_pred = y_pred.cpu().long()
    if nc is None:
        nc: int = max(y_true.max(), y_pred.max())
    iter_nc = range(nc)
    results_dice_list = [.0] * nc
    print(y_true.shape, y_pred.shape)
    for c in iter_nc:
        y_true_nobg = torch.logical_and(y_true[c] == 1, y_true[0] == 0)
        y_pred_nobg = torch.logical_and(y_pred[c] == 1, y_true[0] == 0)

        union = torch.logical_or(y_true_nobg, y_pred_nobg).sum()
        cross_area = torch.logical_and(y_true_nobg, y_pred_nobg).sum()
        # union = torch.logical_or(y_true[c] == 1, y_pred[c] == 1).sum()
        # cross_area = torch.logical_and(y_true[c] == 1, y_pred[c] == 1).sum()
        dice = 2 * cross_area / (union + cross_area)
        results_dice_list[c] = dice if not torch.isnan(dice) else 0
    final_list = list(map(float, results_dice_list))


    return final_list


class ConfusionMatrixBuilder(object):
    pred_seq: list[int] = list()
    gt_seq: list[int] = list()
    slice_seq: dict[str, list[int]] = dict()
    nc: int = None
    def __init__(
            self,
            pred_seq: Optional[List[int] | torch.Tensor | MetaTensor] = None,
            gt_seq: Optional[List[int] | torch.Tensor | MetaTensor] = None,
            nc: Optional[int] = None
    ):
        if pred_seq is not None and gt_seq is not None:
            self.pred_seq = ConfusionMatrixBuilder.preprocess_value(pred_seq)
            self.gt_seq = ConfusionMatrixBuilder.preprocess_value(gt_seq)
        self.nc = nc if nc is not None else 0
        self.slice_seq['pred'] = list()
        self.slice_seq['gt'] = list()

    @classmethod
    def preprocess_value(cls, value, value_is_onehot: bool = True):
        if torch.is_tensor(value):
            value = value.cpu().long()
        if isinstance(value, MetaTensor):
            value = value.as_tensor().cpu().long()

        if value_is_onehot:
            value = torch.argmax(value, dim=0)
        value: list[int] = list(map(int, value.view(-1)))
        return value
    @torch.no_grad()
    def update(self, key, value, value_is_onehot=True, new_slice=False):
        key = key.lower()
        assert key in ['pred', 'gt', 'true', 'truth'], f'your key is: {key}, the key must belong to ["pred", "gt", "true", "truth"]'

        if torch.is_tensor(value):
            value = value.cpu().long()
        if isinstance(value, MetaTensor):
            value = value.as_tensor().cpu().long()

        if value_is_onehot:
            value = torch.argmax(value, dim=0)
        value: list[int] = list(map(int, value.view(-1)))

        if key in ['pred']:
            self.pred_seq.extend(value)

        if key in ['gt', 'true', 'truth']:
            self.gt_seq.extend(value)

    @property
    def confusion_matrix(self):
        nc: int = max(max(self.pred_seq), max(self.gt_seq), self.nc)
        if nc != self.nc:
            logging.warning(f'User setting nc value is {self.nc} smaller than expected nc: {nc}')
        cm = np.zeros((nc, nc), dtype=np.float32)

        for pred_digit, gt_digit in zip(self.pred_seq, self.gt_seq):
            cm[pred_digit, gt_digit] += 1

        return cm
