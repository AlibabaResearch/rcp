import os.path as osp
from typing import TYPE_CHECKING, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.ops.utils import convert_pointclouds_to_tensor, get_point_covariances
from pytorch3d.ops import knn_points
if TYPE_CHECKING:
    from pytorch3d.structures import Pointclouds

from lib.pointnet2 import pointnet2_utils as pointutils

def get_batch_2d_flow(pc1, pc2, predicted_pc2, paths):
    if 'KITTI' in paths[0] or 'kitti' in paths[0]:
        focallengths = []
        cxs = []
        cys = []
        constx = []
        consty = []
        constz = []
        for path in paths:
            fname = osp.split(path)[-1]
            calib_path = osp.join(
                osp.dirname(__file__),
                'calib_cam_to_cam',
                fname + '.txt')
            with open(calib_path) as fd:
                lines = fd.readlines()
                P_rect_left = \
                    np.array([float(item) for item in
                              [line for line in lines if line.startswith('P_rect_02')][0].split()[1:]],
                             dtype=np.float32).reshape(3, 4)
                focallengths.append(-P_rect_left[0, 0])
                cxs.append(P_rect_left[0, 2])
                cys.append(P_rect_left[1, 2])
                constx.append(P_rect_left[0, 3])
                consty.append(P_rect_left[1, 3])
                constz.append(P_rect_left[2, 3])
        focallengths = np.array(focallengths)[:, None, None]
        cxs = np.array(cxs)[:, None, None]
        cys = np.array(cys)[:, None, None]
        constx = np.array(constx)[:, None, None]
        consty = np.array(consty)[:, None, None]
        constz = np.array(constz)[:, None, None]

        px1, py1 = project_3d_to_2d(pc1, f=focallengths, cx=cxs, cy=cys,
                                    constx=constx, consty=consty, constz=constz)
        px2, py2 = project_3d_to_2d(predicted_pc2, f=focallengths, cx=cxs, cy=cys,
                                    constx=constx, consty=consty, constz=constz)
        px2_gt, py2_gt = project_3d_to_2d(pc2, f=focallengths, cx=cxs, cy=cys,
                                          constx=constx, consty=consty, constz=constz)
    else:
        px1, py1 = project_3d_to_2d(pc1)
        px2, py2 = project_3d_to_2d(predicted_pc2)
        px2_gt, py2_gt = project_3d_to_2d(pc2)

    flow_x = px2 - px1
    flow_y = py2 - py1

    flow_x_gt = px2_gt - px1
    flow_y_gt = py2_gt - py1

    flow_pred = np.concatenate((flow_x[..., None], flow_y[..., None]), axis=-1)
    flow_gt = np.concatenate((flow_x_gt[..., None], flow_y_gt[..., None]), axis=-1)
    return flow_pred, flow_gt



def project_3d_to_2d(pc, f=-1050., cx=479.5, cy=269.5, constx=0, consty=0, constz=0):
    x = (pc[..., 0] * f + cx * pc[..., 2] + constx) / (pc[..., 2] + constz)
    y = (pc[..., 1] * f + cy * pc[..., 2] + consty) / (pc[..., 2] + constz)

    return x, y


def estimate_pointcloud_normals(
    pointclouds: Union[torch.Tensor, "Pointclouds"],
    neighborhood_size: int = 50,
    disambiguate_directions: bool = True,
) -> torch.Tensor:
    """
    Estimates the normals of a batch of `pointclouds`.

    The function uses `estimate_pointcloud_local_coord_frames` to estimate
    the normals. Please refer to that function for more detailed information.

    Args:
      **pointclouds**: Batch of 3-dimensional points of shape
        `(minibatch, num_point, 3)` or a `Pointclouds` object.
      **neighborhood_size**: The size of the neighborhood used to estimate the
        geometry around each point.
      **disambiguate_directions**: If `True`, uses the algorithm from [1] to
        ensure sign consistency of the normals of neigboring points.

    Returns:
      **normals**: A tensor of normals for each input point
        of shape `(minibatch, num_point, 3)`.
        If `pointclouds` are of `Pointclouds` class, returns a padded tensor.

    References:
      [1] Tombari, Salti, Di Stefano: Unique Signatures of Histograms for
      Local Surface Description, ECCV 2010.
    """

    curvatures, local_coord_frames = estimate_pointcloud_local_coord_frames(
        pointclouds,
        neighborhood_size=neighborhood_size,
        disambiguate_directions=disambiguate_directions,
    )

    # the normals correspond to the first vector of each local coord frame
    normals = local_coord_frames[:, :, :, 0]

    return normals


def estimate_pointcloud_local_coord_frames(
    pointclouds: Union[torch.Tensor, "Pointclouds"],
    neighborhood_size: int = 50,
    disambiguate_directions: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimates the principal directions of curvature (which includes normals)
    of a batch of `pointclouds`.

    The algorithm first finds `neighborhood_size` nearest neighbors for each
    point of the point clouds, followed by obtaining principal vectors of
    covariance matrices of each of the point neighborhoods.
    The main principal vector corresponds to the normals, while the
    other 2 are the direction of the highest curvature and the 2nd highest
    curvature.

    Note that each principal direction is given up to a sign. Hence,
    the function implements `disambiguate_directions` switch that allows
    to ensure consistency of the sign of neighboring normals. The implementation
    follows the sign disabiguation from SHOT descriptors [1].

    The algorithm also returns the curvature values themselves.
    These are the eigenvalues of the estimated covariance matrices
    of each point neighborhood.

    Args:
      **pointclouds**: Batch of 3-dimensional points of shape
        `(minibatch, num_point, 3)` or a `Pointclouds` object.
      **neighborhood_size**: The size of the neighborhood used to estimate the
        geometry around each point.
      **disambiguate_directions**: If `True`, uses the algorithm from [1] to
        ensure sign consistency of the normals of neigboring points.

    Returns:
      **curvatures**: The three principal curvatures of each point
        of shape `(minibatch, num_point, 3)`.
        If `pointclouds` are of `Pointclouds` class, returns a padded tensor.
      **local_coord_frames**: The three principal directions of the curvature
        around each point of shape `(minibatch, num_point, 3, 3)`.
        The principal directions are stored in columns of the output.
        E.g. `local_coord_frames[i, j, :, 0]` is the normal of
        `j`-th point in the `i`-th pointcloud.
        If `pointclouds` are of `Pointclouds` class, returns a padded tensor.

    References:
      [1] Tombari, Salti, Di Stefano: Unique Signatures of Histograms for
      Local Surface Description, ECCV 2010.
    """

    points_padded, num_points = convert_pointclouds_to_tensor(pointclouds)

    ba, N, dim = points_padded.shape
    if dim != 3:
        raise ValueError(
            "The pointclouds argument has to be of shape (minibatch, N, 3)"
        )

    if (num_points <= neighborhood_size).any():
        raise ValueError(
            "The neighborhood_size argument has to be"
            + " >= size of each of the point clouds."
        )

    # undo global mean for stability
    # TODO: replace with tutil.wmean once landed
    pcl_mean = points_padded.sum(1) / num_points[:, None]
    points_centered = points_padded - pcl_mean[:, None, :]

    # get the per-point covariance and nearest neighbors used to compute it
    cov, knns = get_point_covariances(points_centered, num_points, neighborhood_size)

    # get the local coord frames as principal directions of
    # the per-point covariance
    # this is done with torch.symeig, which returns the
    # eigenvectors (=principal directions) in an ascending order of their
    # corresponding eigenvalues, while the smallest eigenvalue's eigenvector
    # corresponds to the normal direction
    # curvatures, local_coord_frames = torch.symeig(cov, eigenvectors=True)
    
    curvatures, local_coord_frames = torch.symeig(cov.detach().cpu(), eigenvectors=True)
    curvatures, local_coord_frames = curvatures.cuda(), local_coord_frames.cuda()

    # disambiguate the directions of individual principal vectors
    if disambiguate_directions:
        # disambiguate normal
        n = _disambiguate_vector_directions(
            points_centered, knns, local_coord_frames[:, :, :, 0]
        )
        # disambiguate the main curvature
        z = _disambiguate_vector_directions(
            points_centered, knns, local_coord_frames[:, :, :, 2]
        )
        # the secondary curvature is just a cross between n and z
        y = torch.cross(n, z, dim=2)
        # cat to form the set of principal directions
        local_coord_frames = torch.stack((n, y, z), dim=3)

    return curvatures, local_coord_frames


def _disambiguate_vector_directions(pcl, knns, vecs):
    """
    Disambiguates normal directions according to [1].

    References:
      [1] Tombari, Salti, Di Stefano: Unique Signatures of Histograms for
      Local Surface Description, ECCV 2010.
    """
    # parse out K from the shape of knns
    K = knns.shape[2]
    # the difference between the mean of each neighborhood and
    # each element of the neighborhood
    df = knns - pcl[:, :, None]
    # projection of the difference on the principal direction
    proj = (vecs[:, :, None] * df).sum(3)
    # check how many projections are positive
    n_pos = (proj > 0).type_as(knns).sum(2, keepdim=True)
    # flip the principal directions where number of positive correlations
    flip = (n_pos < (0.5 * K)).type_as(knns)
    vecs = (1.0 - 2.0 * flip) * vecs
    return vecs



def estimate_pointcloud_normals_from_two_pcds(
    pcd1, pcd2,
    neighborhood_size: int = 50,
    disambiguate_directions: bool = True,
    radius=None
) -> torch.Tensor:
    """
    Estimates the normals of a batch of `pointclouds`.

    The function uses `estimate_pointcloud_local_coord_frames` to estimate
    the normals. Please refer to that function for more detailed information.

    Args:
      **pointclouds**: Batch of 3-dimensional points of shape
        `(minibatch, num_point, 3)` or a `Pointclouds` object.
      **neighborhood_size**: The size of the neighborhood used to estimate the
        geometry around each point.
      **disambiguate_directions**: If `True`, uses the algorithm from [1] to
        ensure sign consistency of the normals of neigboring points.

    Returns:
      **normals**: A tensor of normals for each input point
        of shape `(minibatch, num_point, 3)`.
        If `pointclouds` are of `Pointclouds` class, returns a padded tensor.

    References:
      [1] Tombari, Salti, Di Stefano: Unique Signatures of Histograms for
      Local Surface Description, ECCV 2010.
    """

    curvatures, local_coord_frames, knns, knns_idx = estimate_pointcloud_local_coord_frames_from_two_pcds(
        pcd1, pcd2,
        neighborhood_size=neighborhood_size,
        disambiguate_directions=disambiguate_directions,
        radius=radius
    )

    # # the normals correspond to the first vector of each local coord frame
    # normals = local_coord_frames[:, :, :, 0]

    return local_coord_frames, knns, knns_idx

def get_point_covariances_from_two_pcds(
    pcd1, pcd2,
    num_points_per_cloud: int,
    neighborhood_size: int,
    radius: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the per-point covariance matrices by of the 3D locations of
    K-nearest neighbors of each point.

    Args:
        **points_padded**: Input point clouds as a padded tensor
            of shape `(minibatch, num_points, dim)`.
        **num_points_per_cloud**: Number of points per cloud
            of shape `(minibatch,)`.
        **neighborhood_size**: Number of nearest neighbors for each point
            used to estimate the covariance matrices.

    Returns:
        **covariances**: A batch of per-point covariance matrices
            of shape `(minibatch, dim, dim)`.
        **k_nearest_neighbors**: A batch of `neighborhood_size` nearest
            neighbors for each of the point cloud points
            of shape `(minibatch, num_points, neighborhood_size, dim)`.
    """
    # get K nearest neighbor idx for each point in the point cloud
    # k_nearest_neighbors = knn_points(
    #     pcd1,
    #     pcd2,
    #     lengths1=num_points_per_cloud,
    #     lengths2=num_points_per_cloud,
    #     K=neighborhood_size,
    #     return_nn=True,
    # ).knn

    dist, idx = pointutils.knn(neighborhood_size, pcd1, pcd2)
    if radius is not None:
        tmp_idx = idx[:, :, 0].unsqueeze(2).repeat(1, 1, neighborhood_size).to(idx.device)
        idx[dist > radius] = tmp_idx[dist > radius]
    k_nearest_neighbors = pointutils.grouping_operation(pcd2.permute(0, 2, 1).contiguous(), idx).permute(0, 2, 3, 1)  # [B, N, K, 3]
        
    # obtain the mean of the neighborhood
    pt_mean = k_nearest_neighbors.mean(2, keepdim=True)
    # compute the diff of the neighborhood and the mean of the neighborhood
    central_diff = k_nearest_neighbors - pt_mean
    # per-nn-point covariances
    per_pt_cov = central_diff.unsqueeze(4) * central_diff.unsqueeze(3)
    # per-point covariances
    covariances = per_pt_cov.mean(2)
    return covariances, k_nearest_neighbors, idx


def estimate_pointcloud_local_coord_frames_from_two_pcds(
    pcd1, pcd2,
    neighborhood_size: int = 50,
    disambiguate_directions: bool = True,
    radius=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """s
    Estimates the principal directions of curvature (which includes normals)
    of a batch of `pointclouds`.

    The algorithm first finds `neighborhood_size` nearest neighbors for each
    point of the point clouds, followed by obtaining principal vectors of
    covariance matrices of each of the point neighborhoods.
    The main principal vector corresponds to the normals, while the
    other 2 are the direction of the highest curvature and the 2nd highest
    curvature.

    Note that each principal direction is given up to a sign. Hence,
    the function implements `disambiguate_directions` switch that allows
    to ensure consistency of the sign of neighboring normals. The implementation
    follows the sign disabiguation from SHOT descriptors [1].

    The algorithm also returns the curvature values themselves.
    These are the eigenvalues of the estimated covariance matrices
    of each point neighborhood.

    Args:
      **pointclouds**: Batch of 3-dimensional points of shape
        `(minibatch, num_point, 3)` or a `Pointclouds` object.
      **neighborhood_size**: The size of the neighborhood used to estimate the
        geometry around each point.
      **disambiguate_directions**: If `True`, uses the algorithm from [1] to
        ensure sign consistency of the normals of neigboring points.

    Returns:
      **curvatures**: The three principal curvatures of each point
        of shape `(minibatch, num_point, 3)`.
        If `pointclouds` are of `Pointclouds` class, returns a padded tensor.
      **local_coord_frames**: The three principal directions of the curvature
        around each point of shape `(minibatch, num_point, 3, 3)`.
        The principal directions are stored in columns of the output.
        E.g. `local_coord_frames[i, j, :, 0]` is the normal of
        `j`-th point in the `i`-th pointcloud.
        If `pointclouds` are of `Pointclouds` class, returns a padded tensor.

    References:
      [1] Tombari, Salti, Di Stefano: Unique Signatures of Histograms for
      Local Surface Description, ECCV 2010.
    """
    pcd1, num_points = convert_pointclouds_to_tensor(pcd1)
    pcd2, _ = convert_pointclouds_to_tensor(pcd2)

    ba, N, dim = pcd1.shape
    if dim != 3:
        raise ValueError(
            "The pointclouds argument has to be of shape (minibatch, N, 3)"
        )

    if (num_points <= neighborhood_size).any():
        raise ValueError(
            "The neighborhood_size argument has to be"
            + " >= size of each of the point clouds."
        )

    # undo global mean for stability
    # TODO: replace with tutil.wmean once landed
    pcl_mean = pcd2.sum(1) / num_points[:, None] #[B, 3]
    pcd2_centered = pcd2 - pcl_mean[:, None, :]
    pcd1_centered = pcd1 - pcl_mean[:, None, :]

    # get the per-point covariance and nearest neighbors used to compute it
    cov, knns, knns_idx = get_point_covariances_from_two_pcds(pcd1_centered, pcd2_centered, num_points, neighborhood_size, radius)
    
    # get the local coord frames as principal directions of
    # the per-point covariance
    # this is done with torch.symeig, which returns the
    # eigenvectors (=principal directions) in an ascending order of their
    # corresponding eigenvalues, while the smallest eigenvalue's eigenvector
    # corresponds to the normal direction
    # curvatures, local_coord_frames = torch.symeig(cov, eigenvectors=True)
    
    curvatures, local_coord_frames = torch.symeig(cov.detach().cpu(), eigenvectors=True)
    curvatures, local_coord_frames = curvatures.cuda(), local_coord_frames.cuda()

    # disambiguate the directions of individual principal vectors
    if disambiguate_directions:
        # disambiguate normal
        n = _disambiguate_vector_directions(
            pcd2_centered, knns, local_coord_frames[:, :, :, 0]
        )
        # disambiguate the main curvature
        z = _disambiguate_vector_directions(
            pcd2_centered, knns, local_coord_frames[:, :, :, 2]
        )
        # the secondary curvature is just a cross between n and z
        y = torch.cross(n, z, dim=2)
        # cat to form the set of principal directions
        local_coord_frames = torch.stack((n, y, z), dim=3)

    # knn:[B, N, K, 3]
    return curvatures, local_coord_frames, knns + pcl_mean[:, None, None, :], knns_idx



def square_dists(points1, points2):
    '''
    Calculate square dists between two group points
    :param points1: shape=(B, N, C)
    :param points2: shape=(B, M, C)
    :return:
    '''
    B, N, C = points1.shape
    _, M, _ = points2.shape
    dists = torch.sum(torch.pow(points1, 2), dim=-1).view(B, N, 1) + \
            torch.sum(torch.pow(points2, 2), dim=-1).view(B, 1, M)
    dists -= 2 * torch.matmul(points1, points2.permute(0, 2, 1))
    return dists.float()


def batch_transform(batch_pc, batch_R, batch_t=None):
    '''

    :param batch_pc: shape=(B, N, 3)
    :param batch_R: shape=(B, 3, 3)
    :param batch_t: shape=(B, 3)
    :return: shape(B, N, 3)
    '''
    transformed_pc = torch.matmul(batch_pc, batch_R.permute(0, 2, 1).contiguous())
    if batch_t is not None:
        transformed_pc = transformed_pc + torch.unsqueeze(batch_t, 1)
    return transformed_pc


def batch_quat2mat(batch_quat):
    '''

    :param batch_quat: shape=(B, 4)
    :return:
    '''
    w, x, y, z = batch_quat[:, 0], batch_quat[:, 1], batch_quat[:, 2], \
                 batch_quat[:, 3]
    device = batch_quat.device
    B = batch_quat.size()[0]
    R = torch.zeros(dtype=torch.float, size=(B, 3, 3)).to(device)
    R[:, 0, 0] = 1 - 2 * y * y - 2 * z * z
    R[:, 0, 1] = 2 * x * y - 2 * z * w
    R[:, 0, 2] = 2 * x * z + 2 * y * w
    R[:, 1, 0] = 2 * x * y + 2 * z * w
    R[:, 1, 1] = 1 - 2 * x * x - 2 * z * z
    R[:, 1, 2] = 2 * y * z - 2 * x * w
    R[:, 2, 0] = 2 * x * z - 2 * y * w
    R[:, 2, 1] = 2 * y * z + 2 * x * w
    R[:, 2, 2] = 1 - 2 * x * x - 2 * y * y
    return R

def batch_mat2quat(mat):
    """
    [B, 4, 4]
    [B, 4]
    """
    w = torch.sqrt(mat[:, 0, 0] + mat[:, 1, 1] + mat[:, 2, 2] + 1 + 1e-8) / 2
    x = (mat[:, 2, 1] - mat[:, 1, 2]) / (4 * w + 1e-8)
    y = (mat[:, 0, 2] - mat[:, 2, 0]) / (4 * w + 1e-8)
    z = (mat[:, 1, 0] - mat[:, 0, 1]) / (4 * w + 1e-8)
    return torch.stack([w, x, y, z], dim=1)


def angle(v1: torch.Tensor, v2: torch.Tensor):
    """Compute angle between 2 vectors

    For robustness, we use the same formulation as in PPFNet, i.e.
        angle(v1, v2) = atan2(cross(v1, v2), dot(v1, v2)).
    This handles the case where one of the vectors is 0.0, since torch.atan2(0.0, 0.0)=0.0

    Args:
        v1: (B, *, 3)
        v2: (B, *, 3)

    Returns:

    """

    cross_prod = torch.stack([v1[..., 1] * v2[..., 2] - v1[..., 2] * v2[..., 1],
                              v1[..., 2] * v2[..., 0] - v1[..., 0] * v2[..., 2],
                              v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]], dim=-1)
    cross_prod_norm = torch.norm(cross_prod, dim=-1)
    dot_prod = torch.sum(v1 * v2, dim=-1)

    return torch.atan2(cross_prod_norm, dot_prod)



def quaternion_to_matrix_torch(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion_torch(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(*batch_dim, 9), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(q_abs.new_tensor(0.1)))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
].reshape(*batch_dim, 4)
    

def inv_R_t(R, t):
    inv_R = R.permute(0, 2, 1).contiguous()
    inv_t = - inv_R @ t[..., None]
    return inv_R, torch.squeeze(inv_t, -1)


def fps(xyz, M):
    '''
    Sample M points from points according to farthest point sampling (FPS) algorithm.
    :param xyz: shape=(B, N, 3)
    :return: inds: shape=(B, M)
    '''
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(size=(B, M), dtype=torch.long).to(device)
    dists = torch.ones(B, N).to(device) * 1e10
    inds = torch.randint(0, N, size=(B, ), dtype=torch.long).to(device)
    batchlists = torch.arange(0, B, dtype=torch.long).to(device)
    for i in range(M):
        centroids[:, i] = inds
        cur_point = xyz[batchlists, inds, :] # (B, 3)
        cur_dist = torch.squeeze(square_dists(torch.unsqueeze(cur_point, 1), xyz), dim=1)
        dists[cur_dist < dists] = cur_dist[cur_dist < dists]
        inds = torch.max(dists, dim=1)[1]
    return centroids


def gather_points(points, inds):
    '''
    :param points: shape=(B, N, C)
    :param inds: shape=(B, M) or shape=(B, M, K)
    :return: sampling points: shape=(B, M, C) or shape=(B, M, K, C)
    '''
    device = points.device
    B, N, C = points.shape
    inds_shape = list(inds.shape)
    inds_shape[1:] = [1] * len(inds_shape[1:])
    repeat_shape = list(inds.shape)
    repeat_shape[0] = 1
    batchlists = torch.arange(0, B, dtype=torch.long).to(device).reshape(inds_shape).repeat(repeat_shape)
    return points[batchlists, inds, :]


def ball_query(xyz, new_xyz, radius, K, rt_density=False):
    '''

    :param xyz: shape=(B, N, 3)
    :param new_xyz: shape=(B, M, 3)
    :param radius: int
    :param K: int, an upper limit samples
    :return: shape=(B, M, K)
    '''
    device = xyz.device
    B, N, C = xyz.shape
    M = new_xyz.shape[1]
    grouped_inds = torch.arange(0, N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, M, 1)
    dists = square_dists(new_xyz, xyz)
    grouped_inds[dists > radius ** 2] = N
    if rt_density:
        density = torch.sum(grouped_inds < N, dim=-1)
        density = density / N
    grouped_inds = torch.sort(grouped_inds, dim=-1)[0][:, :, :K]
    grouped_min_inds = grouped_inds[:, :, 0:1].repeat(1, 1, K)
    grouped_inds[grouped_inds == N] = grouped_min_inds[grouped_inds == N]
    if rt_density:
        return grouped_inds, density
    return grouped_inds


def sample_and_group(xyz, points, M, radius, K, use_xyz=True, rt_density=False):
    '''
    :param xyz: shape=(B, N, 3)
    :param points: shape=(B, N, C)
    :param M: int
    :param radius:float
    :param K: int
    :param use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    :return: new_xyz, shape=(B, M, 3); new_points, shape=(B, M, K, C+3);
             group_inds, shape=(B, M, K); grouped_xyz, shape=(B, M, K, 3)
    '''
    if M < 0:
        new_xyz = xyz
    else:
        new_xyz = gather_points(xyz, fps(xyz, M))
    if rt_density:
        grouped_inds, density = ball_query(xyz, new_xyz, radius, K,
                                           rt_density=True)
    else:
        grouped_inds = ball_query(xyz, new_xyz, radius, K, rt_density=False)
    grouped_xyz = gather_points(xyz, grouped_inds)
    grouped_xyz -= torch.unsqueeze(new_xyz, 2).repeat(1, 1, K, 1)
    if points is not None:
        grouped_points = gather_points(points, grouped_inds)
        if use_xyz:
            new_points = torch.cat((grouped_xyz.float(), grouped_points.float()), dim=-1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz
    if rt_density:
        return new_xyz, new_points, grouped_inds, grouped_xyz, density
    return new_xyz, new_points, grouped_inds, grouped_xyz


def weighted_icp(src, tgt, weights, _EPS = 1e-8):
    """Compute rigid transforms between two point sets

    Args:
        src (torch.Tensor): (B, M, 3) points
        tgt (torch.Tensor): (B, M, 3) points
        weights (torch.Tensor): (B, M)

    Returns:
        R, t, transformed_src: (B, 3, 3), (B, 3), (B, M, 3)

    Modified from open source code:
        https://github.com/yewzijian/RPMNet/blob/master/src/models/rpmnet.py
    """
    weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + _EPS)
    centroid_src = torch.sum(src * weights_normalized, dim=1)
    centroid_tgt = torch.sum(tgt * weights_normalized, dim=1)
    src_centered = src - centroid_src[:, None, :]
    tgt_centered = tgt - centroid_tgt[:, None, :]
    cov = src_centered.transpose(-2, -1) @ (tgt_centered * weights_normalized)

    # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
    # and choose based on determinant to avoid flips
    u, s, v = torch.svd(cov, some=False, compute_uv=True)
    rot_mat_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[:, :, 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(-1, -2)
    rot_mat = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
    assert torch.all(torch.det(rot_mat) > 0)

    # Compute translation (uncenter centroid)
    translation = -rot_mat @ centroid_src[:, :, None] + centroid_tgt[:, :, None]
    translation = torch.squeeze(translation, -1)
    transformed_src = batch_transform(src, rot_mat, translation)
    return rot_mat, translation, transformed_src


def weighted_icp_flow(src_raw, flow, overlap, topk):
    """
    src_raw: [B, N, 3]
    flow: [B, N, 3]
    overlap: [B, N]
    return:
        T: [B, 3, 4]
        src_t: [B, N, 3]
        
    """
    assert topk <= overlap.shape[1]
    ol_score_raw, ol_inds_raw = torch.sort(overlap, dim=-1, descending=True)
    ol_inds = ol_inds_raw[:, :topk] #[B, K]
    ol_score= ol_score_raw[:, :topk] #[B, K]

    src_ol = gather_points(src_raw, ol_inds)
    flow_ol = gather_points(flow, ol_inds)
    
    R, t, _ = weighted_icp(src_ol, src_ol + flow_ol, weights=ol_score)

    src_t = batch_transform(src_raw, R, t)

    return torch.cat([R, t.unsqueeze(2)], dim=2), src_t
