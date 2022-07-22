import torch
# from pytorch_lightning.metrics import TensorMetric
# from pytorch_lightning import metrics
from typing import Any, Optional
from losses.supervised_losses import *
from losses.unsupervised_losses import *
from losses.common_losses import *


class EPE3D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
        epe3d = torch.norm(pred_flow - gt_flow, dim=2).mean()
        return epe3d

class Acc3DR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
        l2_norm = torch.norm(pred_flow - gt_flow, dim=2)
        sf_norm = torch.norm(gt_flow, dim=2)
        relative_err = l2_norm / (sf_norm + 1e-4)
        acc3d_relax = (torch.logical_or(l2_norm < 0.1, relative_err < 0.1)).float().mean()
        return acc3d_relax

class Acc3DS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
        l2_norm = torch.norm(pred_flow - gt_flow, dim=2)
        sf_norm = torch.norm(gt_flow, dim=2)
        relative_err = l2_norm / (sf_norm + 1e-4)
        acc3d_strict = (torch.logical_or(l2_norm < 0.05, relative_err < 0.05)).float().mean()
        return acc3d_strict

class EPE3DOutliers(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
        l2_norm = torch.norm(pred_flow - gt_flow, dim=2)
        sf_norm = torch.norm(gt_flow, dim=2)
        relative_err = l2_norm / (sf_norm + 1e-4)
        epe3d_outliers = (torch.logical_or(l2_norm > 0.3, relative_err > 0.1)).float().mean()
        return epe3d_outliers

class SupervisedL1LossMetric(torch.nn.Module):
    def __init__(self):
        super(SupervisedL1LossMetric, self).__init__()
        self.loss = SupervisedL1Loss()
    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
        loss_metric = self.loss(pc_source, pc_target, pred_flow, gt_flow)
        return loss_metric


class SmoothnessLossMetric(torch.nn.Module):
    def __init__(self, smoothness_loss_params):
        super(SmoothnessLossMetric, self).__init__()
        self.loss = SmoothnessLoss(**smoothness_loss_params)
    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
        loss_metric = self.loss(pc_source, pred_flow)
        return loss_metric

class ChamferLossMetric(torch.nn.Module):
    def __init__(self, chamfer_loss_params):
        super(ChamferLossMetric, self).__init__()
        self.loss = ChamferLoss(**chamfer_loss_params)
    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
        loss_metric = self.loss(pc_source, pc_target, pred_flow)
        return loss_metric


class SceneFlowMetrics():
    """
    An object of relevant metrics for scene flow.
    """

    def __init__(self, split: str, loss_params: dict, reduce_op: Optional[Any] = None):
        """
        Initializes a dictionary of metrics for scene flow
        keep reduction as 'none' to allow metrics computation per sample.

        Arguments:
            split : a string with split type, should be used to allow logging of same metrics for different aplits
            loss_params: loss configuration dictionary
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                        Defaults to sum.
        """

        self.metrics = {
            split + '_epe3d': EPE3D(),

        }
        if loss_params['loss_type'] == 'sv_l1_reg':
            self.metrics[f'{split}_data_loss'] = SupervisedL1LossMetric()
            self.metrics[f'{split}_smoothness_loss'] = SmoothnessLossMetric(loss_params['smoothness_loss_params'])

        if loss_params['loss_type'] == 'unsup_l1':
            self.metrics[f'{split}_chamfer_loss'] = ChamferLossMetric(loss_params['chamfer_loss_params'])
            self.metrics[f'{split}_smoothness_loss'] = SmoothnessLossMetric(loss_params['smoothness_loss_params'])

        if split in ['test', 'val']:
            self.metrics[f'{split}_acc3dr'] = Acc3DR()
            self.metrics[f'{split}_acc3ds'] = Acc3DS()
            self.metrics[f'{split}_epe3d_outliers'] = EPE3DOutliers()

    def __call__(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flows: list, gt_flow: torch.Tensor) -> dict:
        """
        Compute and scale the resulting metrics

        Arguments:
            pc_source : a tensor containing source point cloud
            pc_target : a tensor containing target point cloud
            pred_flows : list of tensors containing model's predictions
            gt_flow : a tensor containing ground truth labels

        Return:
            A dictionary of copmuted metrics
        """
        with torch.no_grad():
            result = {}
            for key, metric in self.metrics.items():
                for i, pred_flow in enumerate(pred_flows):
                    val = metric(pc_source, pc_target, pred_flow, gt_flow)
                    result.update({f'{key}_i#{i}': val})

        return result
