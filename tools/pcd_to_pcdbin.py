import os
from pypcd import pypcd
import numpy as np

pcd_path = '/home/seu/Documents/rosbags/some_peoples/'
pcd_bin_path = '/home/seu/workspace/lmx_ws/FRNet/data/campus/some_people/'

for filename in os.listdir(pcd_path):
    if filename.endswith('.pcd'):
        points = pypcd.PointCloud.from_path(pcd_path + filename)
        points = np.vstack((points.pc_data['x'],points.pc_data['y'],points.pc_data['z'],points.pc_data['intensity'],points.pc_data['ring'])).T
        points = points[~np.isnan(points).any(axis=1)]
        
        # 将处理后的点云数据保存为.pcd.bin文件
        bin_filename = filename.replace('.pcd', '.pcd.bin')
        bin_full_path = os.path.join(pcd_bin_path, bin_filename)
        points.tofile(bin_full_path)
        print(f"Saved {bin_filename} to {pcd_bin_path}")

