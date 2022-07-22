import torch
from torch.nn import Module, MSELoss, L1Loss
from .common_losses import CurvatureLoss, SmoothnessLoss, curvature, get_loss_weights


class SupervisedL1Loss(Module):
    def __init__(self, **kwargs):
        super(SupervisedL1Loss, self).__init__()
        self.l1_loss = L1Loss()

    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
        return self.l1_loss(pred_flow, gt_flow)


class SupervisedL2Loss(Module):
    def __init__(self, **kwargs):
        super(SupervisedL2Loss, self).__init__()
        self.l2_loss = MSELoss()

    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
        return self.l2_loss(pred_flow, gt_flow)


class SupervisedL1RegLoss(Module):
    def __init__(self, w_data, w_smoothness, smoothness_loss_params, **kwargs):
        super(SupervisedL1RegLoss, self).__init__()
        self.data_loss = L1Loss()
        self.smoothness_loss = SmoothnessLoss(**smoothness_loss_params)
        self.w_data = w_data
        self.w_smoothness = w_smoothness

    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor, gt_flow: torch.Tensor, seq_len=0, i=0) -> torch.Tensor:
        if len(self.w_data) == 1:
            w_data = self.w_data[0]
            w_smoothness = self.w_smoothness[0]
        else:
            assert seq_len > 0
            w_data = get_loss_weights(self.w_data, seq_len, i)[i]
            w_smoothness = get_loss_weights(self.w_smoothness, seq_len, i)[i]

        loss = (w_data * self.data_loss(pred_flow, gt_flow)) + (w_smoothness * self.smoothness_loss(pc_source, pred_flow))
        return loss
    

class SupervisedL1RegLossV1(Module):
    def __init__(self, w_data, w_smoothness, smoothness_loss_params, **kwargs):
        super(SupervisedL1RegLossV1, self).__init__()
        self.data_loss = L1Loss()
        self.smoothness_loss = SmoothnessLoss(**smoothness_loss_params)
        
        curvature_loss_params = kwargs.get("curvature_loss_params", {})
        w_curvature = kwargs.get("w_curvature", [0.1])
        print("CurvatureLoss", curvature_loss_params, w_curvature)
        self.curvature_loss = CurvatureLoss(**curvature_loss_params)
        
        self.w_data = w_data
        self.w_smoothness = w_smoothness
        self.w_curvature = w_curvature

    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor, gt_flow: torch.Tensor, seq_len=0, i=0) -> torch.Tensor:
        """
        pc_source: [B, N, 3]
        """
        if len(self.w_data) == 1:
            w_data = self.w_data[0]
            w_smoothness = self.w_smoothness[0]
            w_curvature = self.w_curvature[0]
        else:
            assert seq_len > 0
            w_data = get_loss_weights(self.w_data, seq_len, i)[i]
            w_smoothness = get_loss_weights(self.w_smoothness, seq_len, i)[i]
            w_curvature = get_loss_weights(self.w_curvature, seq_len, i)[i]
        
        loss = (w_data * self.data_loss(pred_flow, gt_flow)) + (w_smoothness * self.smoothness_loss(pc_source, pred_flow)) + \
               (w_curvature / 200 * self.curvature_loss(pc_source, pc_target, pred_flow))

        return loss
    