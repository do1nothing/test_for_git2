from pypcd import pypcd
import numpy as np
import torch
# from mmengine.fileio import get

pcd_path = '/home/seu/workspace/lmx_ws/FRNet/data/campus/gate_01/1627729599.062213421.pcd'
pcd_bin_path = '/home/seu/workspace/lmx_ws/FRNet/data/nuscenes/samples/LIDAR_TOP/n008-2018-08-01-15-34-25-0400__LIDAR_TOP__1533152219048291.pcd.bin'
gt_label = '/home/seu/workspace/lmx_ws/FRNet/work_dirs/frnet-campus_some_people/preds_gate_01/1627729599.062213421.bin'

labels = torch.load(gt_label,map_location=torch.device('cpu')).cpu().numpy()
# import pdb;pdb.set_trace()
points = pypcd.PointCloud.from_path(pcd_path)
points_data = np.vstack((points.pc_data['x'],points.pc_data['y'],points.pc_data['z'],points.pc_data['intensity'],points.pc_data['ring'],points.pc_data['time'])).T
points_data = points_data[~np.isnan(points_data).any(axis=1)]

##筛选所需要的rings
filter_rings = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32]
# filter_rings = [0,1,2]
filter_index = np.ones(len(points_data), dtype=bool)
for num_ring in filter_rings:
    filter_index &= (points_data[:,4] != num_ring)

print(filter_index)
metadata = points.metadata_ori
metadata['points'] = filter_index.sum()
metadata['width'] = filter_index.sum()

filter_points_data = [tuple(row) for row in points_data[filter_index]]
# 定义字段名和类型
fields = ['x', 'y', 'z', 'intensity', 'ring', 'time']
types = ['<f4', '<f4', '<f4', '<f4', '<u2', '<f4']

# 创建记录数组
pc_data = np.array(filter_points_data, dtype=list(zip(fields, types)))
filter_points = pypcd.PointCloud(metadata,pc_data)

pypcd.save_point_cloud_bin(filter_points,'/home/seu/workspace/lmx_ws/FRNet/data/campus/1627729599.062213421_16.pcd')


# pts_bytes = get(pcd_bin_path, backend_args=None)
# points_pcdbin = np.frombuffer(pts_bytes, dtype=np.float32)
# points_pcdbin = points_pcdbin.reshape(-1,5)

# print(points_pcdbin.shape)