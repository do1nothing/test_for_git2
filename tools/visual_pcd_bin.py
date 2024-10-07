import open3d as o3d
import numpy as np
import torch
import glob
import os
from mmengine.fileio import get

# 定义12种自定义颜色 (B, G, R)
class_colors = {
    0: [0, 255, 0], ## green : barrier
    1: [127, 127, 127], 
    2: [255, 0, 0], ## blue : bus
    3: [0, 255, 255], 
    4: [255, 0, 255],
    5: [127, 127, 0], 
    6: [255, 255, 0], 
    7: [0, 127, 127], 
    8: [0, 0, 127], 
    9: [0, 63, 63],
    10: [0, 0, 255], ##red : driveable_surface
    11: [127, 0, 127],
    12: [63, 63, 0],
    13: [63, 0, 63],
    14: [0, 63, 127],
    15: [127, 63, 0],
    16: [255, 127, 0]
    # 17: [0, 127, 255],
    # 18: [127, 255, 0],
    # 19: [255, 0, 127],
    # 20: [0, 127, 63],
    # 21: [127, 63, 127],
    # 22: [63, 127, 0],
    # 23: [255, 63, 0],
    # 24: [63, 0, 255],
    # 25: [127, 0, 255],
    # 26: [0, 255, 127],
    # 27: [127, 255, 127],
    # 28: [255, 127, 127],
    # 29: [0, 255, 63],
    # 30: [127, 0, 63],
    # 31: [255, 63, 127]
}
        
def load_pcd(file_path):
    """读取点云文件"""
    data = np.fromfile(file_path, dtype=np.float32).reshape(-1, 5)[:,[0,1,2,3]]  # 读取XYZI点云
    # import pdb;pdb.set_trace()
    # pts_bytes = get(file_path, backend_args=None)
    # data = np.frombuffer(pts_bytes, dtype=np.float32).reshape(-1, 5)[:,[0,1,2,3]]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:, :3])  # 仅使用XYZ坐标
    return pcd

def load_labels(file_path):
    """读取标签文件"""
    # labels = np.fromfile(file_path, dtype=np.int64, count = -1)
    labels = torch.load(file_path,map_location=torch.device('cpu')).cpu().numpy()
    return labels
    # 读取字节
    # with open(file_path, "rb") as file:
    #     tensor_bytes = file.read()

    # # 将字节转换为NumPy数组
    # # 注意：你需要知道原始数组的形状和数据类型
    # loaded_numpy_array = np.frombuffer(tensor_bytes, dtype=np.int64).reshape((2, 3))
    # return loaded_numpy_array

def convert_bgr_to_rgb(bgr_color):
    return [bgr_color[2], bgr_color[1], bgr_color[0]]

def generate_colors(labels, class_colors):
    colors = np.zeros((len(labels), 3))
    unique_labels = np.unique(labels)
    print(f"Unique labels in the data: {unique_labels}")
    for label, color in class_colors.items():
        if label in unique_labels:
            rgb_color = convert_bgr_to_rgb(color)
            colors[labels == label] = np.array(rgb_color) / 255.0  # 将颜色值标准化到 [0, 1]
    return colors

def visualize_point_cloud_with_labels(pcd, labels, class_colors):
    """可视化带标签的点云"""
    colors = generate_colors(labels, class_colors)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

def vis_from_path(path):
    
    # 使用glob模块查找所有以.bin结尾的文件
    bin_files = glob.glob(os.path.join(path, '*.bin'))
    # 遍历找到的文件
    for file in bin_files:
        print(f'Reading file: {file}')
        gt_name = file.split('/')[-1].split('.bin')[0]
        # print(gt_name)
        # pcd_file_path = '/home/seu/workspace/lmx_ws/FRNet/data/campus/some_people/'+gt_name+'.pcd.bin'
        # labels_file_path = '/home/seu/workspace/lmx_ws/FRNet/work_dirs/frnet-campus_some_people/preds/' + gt_name + '.bin'
        pcd_file_path = '/home/seu/workspace/lmx_ws/FRNet/data/campus/gate_01/'+gt_name+'.pcd.bin'
        labels_file_path = '/home/seu/workspace/lmx_ws/FRNet/work_dirs/frnet-campus_some_people/preds_gate_01/' + gt_name + '.bin'
    

        pcd = load_pcd(pcd_file_path)
    
        # 加载标签
        labels = load_labels(labels_file_path)
        print("points:",len(pcd.points))
        print("labels:",len(labels))
        assert len(pcd.points) == len(labels)

        visualize_point_cloud_with_labels(pcd, labels, class_colors)
    
def vis_from_single():
    
    # pcd_file_path = "/home/seu/workspace/lmx_ws/FRNet/data/nuscenes/samples/LIDAR_TOP/n008-2018-08-01-15-34-25-0400__LIDAR_TOP__1533152646947278.pcd.bin"
    # labels_file_path = "/home/seu/workspace/lmx_ws/FRNet/work_dirs/frnet-nuscenes_seg_test/preds/n008-2018-08-01-15-34-25-0400__LIDAR_TOP__1533152646947278.bin"
    # pcd_file_path = '/home/seu/workspace/lmx_ws/FRNet/data/campus/some_people/1627729602.569065809.pcd.bin' ## 16 rings
    # labels_file_path = '/home/seu/workspace/lmx_ws/FRNet/work_dirs/frnet-campus_some_people/preds/1627729602.569065809.bin'
    pcd_file_path = '/home/seu/workspace/lmx_ws/FRNet/data/campus/gate_01/1627729602.969836235.pcd.bin'
    labels_file_path = '/home/seu/workspace/lmx_ws/FRNet/work_dirs/frnet-campus_some_people/preds_gate_01/1627729602.969836235.bin'
    
    
    
    pcd = load_pcd(pcd_file_path)
    
    # 加载标签
    labels = load_labels(labels_file_path)
    print("points:",len(pcd.points))
    print("labels:",len(labels))
    assert len(pcd.points) == len(labels)

    visualize_point_cloud_with_labels(pcd, labels, class_colors)
        
def main():
    # 指定路径
    path = '/home/seu/workspace/lmx_ws/FRNet/work_dirs/frnet-campus_some_people/preds_gate_01'
    vis_from_path(path)
    # vis_from_single()
    
    

    
    

if __name__ == "__main__":
    main()