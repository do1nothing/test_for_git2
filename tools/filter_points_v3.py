import numpy as np

def rotate_points(points, yaw, center):
    """
    旋转点集，使其与框的坐标系对齐。
    
    :param points: 需要旋转的点集 (n, 3)
    :param yaw: 偏航角
    :param center: 框的中心坐标 (3,)
    :return: 旋转后的点集
    """
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    # 构造旋转矩阵
    rotation_matrix = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw,  cos_yaw, 0],
        [0,        0,       1]
    ])
    
    # 先将点集平移到以框的中心为原点
    points_shifted = points - center
    
    # 旋转点集
    rotated_points = np.dot(points_shifted, rotation_matrix.T)
    
    return rotated_points

def filter_points_in_box(points, box):
    """
    过滤框内的点
    
    :param points: 输入的三维点集 (n, 3)
    :param box: 7维张量 (x, y, z, len_x, len_y, len_z, yaw)
    :return: 仅保留在框内的点
    """
    x, y, z, len_x, len_y, len_z, yaw = box
    center = np.array([x, y, z])
    
    # 将点集旋转到与框对齐的坐标系
    rotated_points = rotate_points(points, yaw, center)
    
    # 获取框的半边长
    half_lengths = np.array([len_x / 2, len_y / 2, len_z / 2])
    
    # 检查哪些点在框的范围内
    within_box_mask = (
        (rotated_points[:, 0] >= -half_lengths[0]) & (rotated_points[:, 0] <= half_lengths[0]) &
        (rotated_points[:, 1] >= -half_lengths[1]) & (rotated_points[:, 1] <= half_lengths[1]) &
        (rotated_points[:, 2] >= -half_lengths[2]) & (rotated_points[:, 2] <= half_lengths[2])
    )
    
    # 返回在框内的点
    points_in_box = points[within_box_mask]
    
    return points_in_box

# 示例输入
points = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [5, 6, 7],
    [8, 9, 10]
])

box = np.array([5, 5, 5, 4, 4, 4, np.pi/4])

# 过滤框内的点
points_in_box = filter_points_in_box(points, box)
print("Points inside the box:", points_in_box)
