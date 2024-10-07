import os
import json
import pickle

# pcd_bin_path = '/home/seu/workspace/lmx_ws/FRNet/data/nuscenes/samples/LIDAR_TOP'
pcd_bin_path = '/home/seu/workspace/lmx_ws/FRNet/data/campus/some_people'

results = {}
results['metainfo'] = {
    'dataset': 'nuscenes', 
    'version': 'v1.0-test'
}
results['data_list'] = []
# results['lidar_points'] = {}
# 
for filename in os.listdir(pcd_bin_path):
    try:
        if filename.endswith('.pcd.bin'):
            # lidar_idx = os.path.splitext(filename)[0]
            pcd_file_path = os.path.join(pcd_bin_path,filename)
            lidar_idx = filename
        
            # 检查 selfmerge_pcd.pcd 文件是否存在
            if not os.path.exists(pcd_file_path):
                print(f'File not found: {pcd_file_path}')
        
## 'metainfo':{'dataset': 'nuscenes', 'version': 'v1.0-test'}
# {'lidar_points': {'lidar_path': 'n008-2018-08-01-16-03-27-0400__LIDAR_TOP__1533153857947444.pcd.bin', 
#                 'num_pts_feats': 5, 'sample_data_token': '47d2f6131ba449f6a03ced4bde2c6dbe'}, 'token': '1b9a789e08bb4b7b89eacb2176c70840'}
       
            # 生成新的 info 列表项
            info_item = {'lidar_points':{
                'lidar_path': lidar_idx,
                'num_pts_feats': 5
            }
            }
        # {'metainfo': {'dataset': 'nuscenes', 'version': 'v1.0-test'}, 
        # 'lidar_points': {'lidar_path': 'n008-2018-08-28-15-47-40-0400__LIDAR_TOP__1535485824897681.pcd.bin', 'num_pts_feats': 5}}
        
            # 添加到结果列表
            results['data_list'].append(info_item)
    
    except KeyError as e:
        print(f'Missing expected key: {e} in item {item}')

# 输出生成的 info 列表
# print(results)

output_path = '/home/seu/workspace/lmx_ws/FRNet/data/campus/some_people.pkl'
with open(output_path,'wb') as f:
    pickle.dump(results,f)
    
print("ok!!!!!!!!!!")


