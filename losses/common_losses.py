import torch
from torch.nn import Module, MSELoss, L1Loss
from lib.pointnet2 import pointnet2_utils as pointutils

RADIUS = 2.5

def get_loss_weights(weights, seq_len, gamma):
    
    if weights is None:
        return [1.0 / seq_len] * seq_len
    
    elif len(weights) == 1:
        return [weights] * seq_len
    
    elif len(weights) == 2:
        w_init, w_base = weights[0], weights[1]
        weights_new = [w_init, w_base]
        for i in range(2, seq_len):
            w = w_base * gamma ** (i - 1)
            weights_new.append(w)
            
        return  [1.0 * e / sum(weights_new) for e in weights_new]
    
    elif len(weights) == seq_len:
        return weights
    
    else:
        raise NotImplementedError
    
class KnnLoss(Module):
    def __init__(self, k, radius, loss_norm, **kwargs):
        super(KnnLoss, self).__init__()
        self.k = k
        self.radius = radius
        self.loss_norm = loss_norm

    def forward(self, pc_source: torch.Tensor, pred_flow: torch.Tensor) -> torch.Tensor:
        flow = pred_flow.permute(0, 2, 1).contiguous()
        dist, idx = pointutils.knn(self.k, pc_source, pc_source)
        tmp_idx = idx[:, :, 0].unsqueeze(2).repeat(1, 1, self.k).to(idx.device)
        idx[dist > self.radius] = tmp_idx[dist > self.radius]
        nn_flow = pointutils.grouping_operation(flow, idx.detach())
        loss = (flow.unsqueeze(3) - nn_flow).norm(p=self.loss_norm, dim=1).mean( dim=-1)
        return loss.mean()


class BallQLoss(Module):
    def __init__(self, k, radius, loss_norm, **kwargs):
        super(BallQLoss, self).__init__()
        self.k = k
        self.radius = radius
        self.loss_norm = loss_norm

    def forward(self, pc_source: torch.Tensor, pred_flow: torch.Tensor) -> torch.Tensor:
        flow = pred_flow.permute(0, 2, 1).contiguous()
        idx = pointutils.ball_query(self.radius, self.k, pc_source, pc_source)
        nn_flow = pointutils.grouping_operation(flow, idx.detach())  # retrieve flow of nn
        loss = (flow.unsqueeze(3) - nn_flow).norm(p=self.loss_norm, dim=1).mean( dim=-1)
        return loss.mean()


class SmoothnessLoss(Module):
    def __init__(self, w_knn, w_ball_q, knn_loss_params, ball_q_loss_params, **kwargs):
        super(SmoothnessLoss, self).__init__()
        self.knn_loss = KnnLoss(**knn_loss_params)
        self.ball_q_loss = BallQLoss(**ball_q_loss_params)
        self.w_knn = w_knn
        self.w_ball_q = w_ball_q

    def forward(self, pc_source: torch.Tensor, pred_flow: torch.Tensor) -> torch.Tensor:
        loss = (self.w_knn * self.knn_loss(pc_source, pred_flow)) + (self.w_ball_q * self.ball_q_loss(pc_source, pred_flow))
        return loss


class ChamferLoss(Module):
    def __init__(self, k, loss_norm, **kwargs):
        super(ChamferLoss, self).__init__()
        self.k = k
        self.loss_norm = loss_norm

    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor) -> torch.Tensor:
        pc_target = pc_target.contiguous()
        pc_target_t = pc_target.permute(0, 2, 1).contiguous()
        pc_pred = (pc_source + pred_flow).contiguous()
        pc_pred_t = pc_pred.permute(0, 2, 1).contiguous()

        _, idx = pointutils.knn(self.k, pc_pred, pc_target)
        nn1 = pointutils.grouping_operation(pc_target_t, idx.detach())
        dist1 = (pc_pred_t.unsqueeze(3) - nn1).norm(p=self.loss_norm, dim=1).mean( dim=-1)  # nn flow consistency
        _, idx = pointutils.knn(self.k, pc_target, pc_pred)
        nn2 = pointutils.grouping_operation(pc_pred_t, idx.detach())
        dist2 = (pc_target_t.unsqueeze(3) - nn2).norm(p=self.loss_norm, dim=1).mean( dim=-1)  # nn flow consistency
        ch_dist = (dist1 + dist2)
        return ch_dist.mean()

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx


def index_points_group(points, knn_idx):
    """
    Input:
        points: input points data, [B, N, C]
        knn_idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointutils.grouping_operation(points_flipped, knn_idx.int()).permute(0, 2, 3, 1)

    return new_points


def curvatureWarp(pc, warped_pc, nsample=10, radius=RADIUS):
    # pc: B 3 N
    assert pc.shape[1] == 3
    warped_pc = warped_pc.permute(0, 2, 1)
    pc = pc.permute(0, 2, 1)
    
    dist, kidx = pointutils.knn(nsample, pc.contiguous(), pc.contiguous()) #(B, N, 10)

    if radius is not None:
        tmp_idx = kidx[:, :, 0].unsqueeze(2).repeat(1, 1, nsample).to(kidx.device)
        kidx[dist > radius] = tmp_idx[dist > radius]

    grouped_pc = index_points_group(warped_pc, kidx)
    pc_curvature = torch.sum(grouped_pc - warped_pc.unsqueeze(2), dim = 2) / 9.0
    return pc_curvature # B N 3


def curvature(pc, nsample=10, radius=RADIUS):
    # pc: B 3 N
    assert pc.shape[1] == 3
    pc = pc.permute(0, 2, 1)
    
    dist, kidx = pointutils.knn(nsample, pc.contiguous(), pc.contiguous()) #(B, N, 10)

    if radius is not None:
        tmp_idx = kidx[:, :, 0].unsqueeze(2).repeat(1, 1, nsample).to(kidx.device)
        kidx[dist > radius] = tmp_idx[dist > radius]

    # sqrdist = square_distance(pc, pc)
    # _, kidx = torch.topk(sqrdist, 10, dim = -1, largest=False, sorted=False) 
    
    grouped_pc = index_points_group(pc, kidx) # B N 10 3
    pc_curvature = torch.sum(grouped_pc - pc.unsqueeze(2), dim = 2) / 9.0
    return pc_curvature # B N 3


def interpolateCurvature(pc1, pc2, pc2_curvature, nsample=5, radius=RADIUS):
    '''
    pc1: B 3 N
    pc2: B 3 M
    pc2_curvature: B 3 M
    '''
    assert pc1.shape[1] == 3
    assert pc2.shape[1] == 3

    B, _, N = pc1.shape
    pc1 = pc1.permute(0, 2, 1)
    pc2 = pc2.permute(0, 2, 1)
    pc2_curvature = pc2_curvature

    dist, kidx = pointutils.knn(nsample, pc1.contiguous(), pc2.contiguous()) #(B, N, 10)

    if radius is not None:
        tmp_idx = kidx[:, :, 0].unsqueeze(2).repeat(1, 1, nsample).to(kidx.device)
        kidx[dist > radius] = tmp_idx[dist > radius]
        
    grouped_pc2_curvature = index_points_group(pc2_curvature, kidx) # B N 5 3
    norm = torch.sum(1.0 / (dist + 1e-8), dim = 2, keepdim = True)
    weight = (1.0 / (dist + 1e-8)) / norm

    inter_pc2_curvature = torch.sum(weight.view(B, N, 5, 1) * grouped_pc2_curvature, dim = 2)
    return inter_pc2_curvature


class CurvatureLoss(Module):
    def __init__(self, **kwargs):
        super(CurvatureLoss, self).__init__()

    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor) -> torch.Tensor:
        """
        pc_source: [B, N, 3]
        """
        pc_source = pc_source.permute(0, 2, 1).contiguous()
        pc_target = pc_target.permute(0, 2, 1).contiguous()
        pred_flow = pred_flow.permute(0, 2, 1).contiguous()
        
        cur_pc1_warp = pc_source + pred_flow
        
        cur_pc2_curvature = curvature(pc_target)  
        moved_pc1_curvature = curvatureWarp(pc_source, cur_pc1_warp)
        #curvature
        inter_pc2_curvature = interpolateCurvature(cur_pc1_warp, pc_target, cur_pc2_curvature)
        curvatureLoss = torch.sum((inter_pc2_curvature - moved_pc1_curvature) ** 2, dim = 2).sum(dim = 1).mean()
        
        return curvatureLoss

if __name__ == "__main__":
    xyz1 = torch.arange(300).reshape(1, 3, 100).float().cuda()
    cur1 = curvature(xyz1)
    print(xyz1.shape)
    print(cur1)