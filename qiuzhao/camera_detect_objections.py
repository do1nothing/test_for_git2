import math


# 计算障碍物相对于原点的极坐标（距离和角度）
def polar_coordinates(x, y):
    distance = math.sqrt(x ** 2 + y ** 2)
    angle = math.degrees(math.atan2(y, x))
    return distance, angle


# 判断障碍物是否在给定视角范围内
def is_within_fov(angle, start_angle, fov):
    end_angle = start_angle + fov
    return start_angle <= angle <= end_angle


# 主函数
def max_detected_obstacles(cameras, obstacles):
    detected_obstacles = set()

    for fov, detection_range in cameras:
        max_count = 0
        best_set = set()

        # 计算每个障碍物的极坐标
        polar_coords = [polar_coordinates(x, y) for x, y in obstacles]

        # 遍历所有障碍物，尝试将它们作为视角的起点
        for i in range(len(polar_coords)):
            start_angle = polar_coords[i][1]
            current_set = set()

            for j in range(len(polar_coords)):
                angle, distance = polar_coords[j]

                # 检查障碍物是否在当前视角范围内，并且距离在探测范围内
                if is_within_fov(angle, start_angle, fov) and distance <= detection_range:
                    current_set.add(obstacles[j])

            # 更新最优结果
            if len(current_set) > max_count:
                max_count = len(current_set)
                best_set = current_set

        detected_obstacles.update(best_set)

    return len(detected_obstacles), detected_obstacles


# 示例数据
cameras = [(50, 20), (150, 20)]  # 每个相机的(fov, detection_range)
obstacles = [(1, 0), (0, 1), (100, 1)]  # 障碍物的坐标

max_count, detected_obstacles = max_detected_obstacles(cameras, obstacles)

print("最多能探测到的障碍物数量:", max_count)
print("探测到的障碍物坐标:", detected_obstacles)