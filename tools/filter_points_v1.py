import numpy as np

def filter_points_v1(points, mask_range):
    mask = [False] * len(points)
    for i, point in enumerate(points):
        x, y, z = point
        if mask_range[3] <= x <= mask_range[0] and mask_range[4] <= y <= mask_range[1] and mask_range[5] <= z <= mask_range[2]:
            mask[i] = True   
    return points[mask]
def filter_points_v2(points, mask_range):
    mask = (
        (points[:, 0] >= mask_range[3]) & (points[:, 0] <= mask_range[0]) &
        (points[:, 1] >= mask_range[4]) & (points[:, 1] <= mask_range[1]) &
        (points[:, 2] >= mask_range[5]) & (points[:, 2] <= mask_range[2])
    )
    return points[~mask]
if __name__ == "__main__":
    points = np.array([[11,13,9],
                       [10,10,10],
                       [8,14,13],
                       [12,11,14],
                       [9,9,9]])
    mask_range = np.array([10,10,10,0,0,0])
    filtered_points_v1 = filter_points_v1(points, mask_range)
    print(f"in the range is \n {filtered_points_v1}")
    filtered_points_v2 = filter_points_v2(points,mask_range)
    print(f"out the range is \n {filtered_points_v2}")