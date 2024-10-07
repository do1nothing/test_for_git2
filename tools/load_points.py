from pypcd import pypcd
import numpy as np
from mmengine.fileio import get

pcd_path = '/home/seu/workspace/lmx_ws/FRNet/data/campus/gate_01/1627729599.062213421.pcd'
pcd_bin_path = '/home/seu/workspace/lmx_ws/FRNet/data/nuscenes/samples/LIDAR_TOP/n008-2018-08-01-15-34-25-0400__LIDAR_TOP__1533152219048291.pcd.bin'

points = pypcd.PointCloud.from_path(pcd_path)
points = np.vstack((points.pc_data['x'],points.pc_data['y'],points.pc_data['z'],points.pc_data['ring'],points.pc_data['intensity'])).T
points = points[~np.isnan(points).any(axis=1)]

pts_bytes = get(pcd_bin_path, backend_args=None)
points_pcdbin = np.frombuffer(pts_bytes, dtype=np.float32)
points_pcdbin = points_pcdbin.reshape(-1,5)

print(points_pcdbin.shape)