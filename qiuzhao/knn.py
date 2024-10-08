import numpy as np

def knn_3d(points, query_point, k):
    """
    寻找空间中距离 query_point 最近的 k 个三维点
    
    :param points: 多个三维点的列表，形状为 (n, 3)
    :param query_point: 查询点，形状为 (3,)
    :param k: 最近邻点的数量
    :return: 最接近的 k 个点的索引列表
    """
    # 计算所有点到查询点的欧氏距离
    distances = np.sqrt(np.sum((points - query_point)**2, axis=1))
    
    # 获取距离最小的 k 个点的索引
    nearest_indices = np.argsort(distances)[:k]
    
    return nearest_indices

# 示例
points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
query_point = np.array([5, 5, 5])
k = 2

nearest_indices = knn_3d(points, query_point, k)
nearest_points = points[nearest_indices]

print("最近的点的索引:", nearest_indices)
print("最近的点:", nearest_points)
