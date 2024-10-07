import numpy as np
##矩阵*坐标=新坐标系下的坐标；坐标*矩阵，原坐标系下旋转后的坐标
##向量不变，坐标轴逆时针；等价于坐标轴不变，向量顺时针。
def rotate_point(point, center, angle):
    """绕给定点中心旋转点"""
    # 将角度转换为弧度
    angle_rad = np.deg2rad(angle)
    # 计算旋转矩阵
    rot_matrix = np.array([
        [np.cos(angle_rad), np.sin(angle_rad)],
        [-np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    # 将点转换为二维数组形式，以进行矩阵乘法
    # point = np.atleast_2d(point)
    # 应用旋转
    rotated_point = np.dot(np.array(rot_matrix), (np.array(point) - np.array(center)).T)
    # 将点移回原点
    rotated_point += center
    return rotated_point

def is_point_in_box(points, box_params):
    # 将points和box_params转换为numpy数组
    points = np.asarray(points)
    box_params = np.asarray(box_params)
    
    # 检查points是否为2维数组，且第二维长度为3（三维坐标）
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be a 2D array with shape (n, 3)")
    
    # 检查box_params是否为1维数组，且长度为7
    if box_params.ndim != 1 or len(box_params) != 7:
        raise ValueError("box_params must be a 1D array with 7 elements")
    
    # 解包box_params
    x_center, y_center, z_center, len_x, len_y, len_z, yaw = box_params
    
    # 初始化mask矩阵，默认为False
    mask = np.zeros(points.shape[0], dtype=bool)
    
    # 遍历每个点，检查是否在框内
    for i, point in enumerate(points):
        # 将点旋转到框的局部坐标系中  这个原点还真不好想
        # 可以优化一下，一次处理一批点
        rotated_point = rotate_point([point[0], point[1]], [x_center, y_center], yaw)
        
        # 检查点是否在框内
        if (abs(rotated_point[0] - x_center) <= len_x / 2) and \
           (abs(rotated_point[1] - y_center) <= len_y / 2) and \
           (point[2] - z_center) >= 0 and (point[2] - z_center) <= len_z:
            mask[i] = True
    
    return mask

if __name__ == "__main__":
    # 示例使用
    points = np.array([
        [5, 12, 5],
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [5, 5, 5]
    ])
    box_params = np.array([5, 5, 5, 10, 10, 10, 45])  # 中心坐标(5,5,5)，长宽高(10,10,10)，偏航角45度

    mask = is_point_in_box(points, box_params)
    print(mask)  # 输出结果，True表示点在框内
    
    '''
    ####旋转后的框 变换到当前坐标系下
    ### 以(5, 5)为新的坐标原点
    box_2d = np.array([[5,5],
                       [5, -5],
                       [-5,5],
                       [-5,-5]])
    ###顺时针旋转45度
    yaw_rad = np.deg2rad(45)
    rotated_box = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad)],
                            [np.sin(yaw_rad), np.cos(yaw_rad)]])
    ##旋转后的以(5,5)为坐标原点后的坐标为
    box_2d_rot = np.dot(rotated_box,box_2d.T).T
    ##以(0,0)为坐标原点的坐标为
    box_2d_rot_0 = box_2d_rot + 5
    print(box_2d_rot_0)
    '''
    
    
    '''
    ###########左乘右乘的验证
    box = np.array([[0, 0, 0],
                    [10,0, 0],
                    [0,10, 0]
                    [10,10,0],
                    [0, 0, 10],
                    [10,0,10],
                    [0, 10, 10],
                    [10,10,10]])
    box_2d = np.array([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1]])*10
    yaw_rad = np.deg2rad(45)
    rotate_matrix = np.array([[np.cos(yaw_rad), np.sin(yaw_rad)],
                             [-np.sin(yaw_rad),np.cos(yaw_rad)]])
    # box_rotate = np.dot(rotate_matrix,box_2d.T)
    box_rotate = np.dot(box_2d,rotate_matrix)
    print(box_rotate)
    '''
    