import numpy as np

def point_cloud_voxelization(points, voxel_size, normalize=True):
    """
    点云体素化
    :param points: 点云数据，形状为 (N, 3)
    :param voxel_size: 体素网格的尺寸
    :param normalize: 是否归一化点云数据
    :return: 体素化后的点云数据
    """
    if normalize:
        # 归一化点云数据
        min_bound = np.min(points, axis=0)
        # max_bound = np.max(points, axis=0)
        # points = (points - min_bound) / (max_bound - min_bound)
        points = (points - min_bound)
    
    # 量化点云数据
    # voxel_indices = (points / voxel_size).astype(np.int32)
    grid_coord = np.floor(points / voxel_size)
    
    # 去重
    unique_voxels = np.unique(grid_coord, axis=0)
    
    return unique_voxels

# 示例用法
points = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.1, 0.2, 0.3],  # 重复点
    [0.7, 0.8, 0.9]
])

voxel_size = 0.1
voxels = point_cloud_voxelization(points, voxel_size)

print("体素化后的点云数据:")
print(voxels)
