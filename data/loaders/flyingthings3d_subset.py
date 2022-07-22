import sys, os
import os.path as osp
import numpy as np
# import open3d as o3d
import torch.utils.data as data

__all__ = ['FlyingThings3DSubset']


class FlyingThings3DSubset(data.Dataset):
    """
    Args:
        train (bool): If True, creates dataset from training set, otherwise creates from test set.
        transform (callable):
        args:
    """
    def __init__(self,
                 train,
                 transform,
                 num_points,
                 data_root,
                 overfit_samples=None,
                 full=True):
        self.root = osp.join(data_root, 'FlyingThings3D_subset_processed_35m')
        self.train = train
        self.transform = transform
        self.num_points = num_points

        self.samples = self.make_dataset(full, overfit_samples)

        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))
        print(f"dataset len:{len(self.samples)}")
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pc1_loaded, pc2_loaded = self.pc_loader(self.samples[index])
        pc1_transformed, pc2_transformed, sf_transformed = self.transform([pc1_loaded, pc2_loaded])
        
        # print(pc1_loaded.shape, pc1_transformed.shape)
        # import open3d as o3d
        # pcd1 = o3d.geometry.PointCloud()
        # pcd1.points = o3d.utility.Vector3dVector(pc1_transformed)
        
        # pcd2 = o3d.geometry.PointCloud()
        # pcd2.points = o3d.utility.Vector3dVector(pc2_transformed)        

        # pcd3 = o3d.geometry.PointCloud()
        # pcd3.points = o3d.utility.Vector3dVector(pc1_loaded)
        
        # pcd4 = o3d.geometry.PointCloud()
        # pcd4.points = o3d.utility.Vector3dVector(pc2_loaded)     
            
        # o3d.io.write_point_cloud("debug1.ply", pcd1)
        # o3d.io.write_point_cloud("debug2.ply", pcd3)
        
        if pc1_transformed is None:
            print('path {} get pc1 is None'.format(self.samples[index]), flush=True)
            index = np.random.choice(range(self.__len__()))
            return self.__getitem__(index)

        pc1_norm = pc1_transformed
        pc2_norm = pc2_transformed
        
        # pcd1 = o3d.geometry.PointCloud()
        # pcd1.points = o3d.utility.Vector3dVector(pc1_transformed)
        # pcd1.colors = o3d.utility.Vector3dVector(pc1_norm)
        
        # pcd2 = o3d.geometry.PointCloud()
        # pcd2.points = o3d.utility.Vector3dVector(pc2_transformed)
        # pcd2.colors = o3d.utility.Vector3dVector(pc2_norm)
        
        # o3d.visualization.draw_geometries([pcd1, pcd2])
        
        # print(pc1_norm.shape, pc2_norm.shape, sf_transformed.shape)
        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # print(f"x: {pc1_norm[:, 0].min()} ~ {pc1_norm[:, 0].max()}, y: {pc1_norm[:, 1].min()} ~ {pc1_norm[:, 1].max()}, z: {pc1_norm[:, 2].min()} ~ {pc1_norm[:, 2].max()}")
        # flow_norm = np.linalg.norm(sf_transformed, axis=1)
        # flow_norm_sort = np.sort(flow_norm)
        # idx_min = flow_norm.argmin()
        # idx_max = flow_norm.argmax()
        # print(f"flow norm min:{sf_transformed[idx_min]}  max:{sf_transformed[idx_max]}", flow_norm_sort[0], flow_norm_sort[-1], 
        #       flow_norm_sort[flow_norm_sort.shape[0]//2], np.mean(flow_norm))
        # fig = plt.figure(figsize=(4,4))
        # ax = fig.add_subplot(131, projection='3d')
        # ax.set_box_aspect(aspect = (1,1,1))
        # ax.scatter(pc1_norm[:, 0], pc1_norm[:, 1], pc1_norm[:, 2])
        # ax = fig.add_subplot(132, projection='3d')
        # ax.set_box_aspect(aspect = (1,1,1))
        # ax.scatter(pc2_norm[:, 0], pc2_norm[:, 1], pc2_norm[:, 2]) 
        # ax = fig.add_subplot(133, projection='3d')  
        # ax.set_box_aspect(aspect = (1,1,1))
        # ax.scatter(sf_transformed[:, 0], sf_transformed[:, 1], sf_transformed[:, 2])     
        # plt.show()
        # xx
        
        return pc1_transformed, pc2_transformed, pc1_norm, pc2_norm, sf_transformed, self.samples[index]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Number of points per point cloud: {}\n'.format(self.num_points)
        fmt_str += '    is training: {}\n'.format(self.train)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def make_dataset(self, full, overfit_samples):
        root = osp.realpath(osp.expanduser(self.root))
        root = osp.join(root, 'train') if (self.train and overfit_samples is None) else osp.join(root, 'val')
        print(root)
        all_paths = os.walk(root)
        useful_paths = sorted([item[0] for item in all_paths if len(item[1]) == 0])
        # try:
        #     if (self.train and overfit_samples is None):
        #         assert (len(useful_paths) == 19640)
        #     else:
        #         assert (len(useful_paths) == 3824)
        # except AssertionError:
        #     print('len(useful_paths) =', len(useful_paths))
        #     sys.exit(1)
        if overfit_samples is not None:
            res_paths = useful_paths[:overfit_samples]
        else:
            if not full:
                res_paths = useful_paths[::4]
            else:
                res_paths = useful_paths

        return res_paths

    def pc_loader(self, path):
        """
        Args:
            path: path to a dir, e.g., home/xiuye/share/data/Driving_processed/35mm_focallength/scene_forwards/slow/0791
        Returns:
            pc1: ndarray (N, 3) np.float32
            pc2: ndarray (N, 3) np.float32
        """
        pc1 = np.load(osp.join(path, 'pc1.npy'))
        pc2 = np.load(osp.join(path, 'pc2.npy'))
        # multiply -1 only for subset datasets
        pc1[..., -1] *= -1
        pc2[..., -1] *= -1
        pc1[..., 0] *= -1
        pc2[..., 0] *= -1
        
        return pc1, pc2
