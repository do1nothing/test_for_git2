import cv2
import numpy as np

# 定义颜色和对应的名称
colors = {
    "green": np.array([0, 255, 0], dtype=np.uint8),
    "gray": np.array([127, 127, 127], dtype=np.uint8),
    "blue": np.array([255, 0, 0], dtype=np.uint8),
    "yellow": np.array([0, 255, 255], dtype=np.uint8),
    "magenta": np.array([255, 0, 255], dtype=np.uint8),
    "olive": np.array([127, 127, 0], dtype=np.uint8),
    "xx": np.array([255, 255, 0], dtype=np.uint8),
    "xy": np.array([0, 127, 127], dtype=np.uint8),
    "indigo": np.array([0, 0, 127], dtype=np.uint8),
    "dark_green": np.array([0, 63, 63], dtype=np.uint8),
    "red": np.array([0, 0, 255], dtype=np.uint8),
    "dark_purple": np.array([127, 0, 127], dtype=np.uint8),
    "dark_olive": np.array([63, 63, 0], dtype=np.uint8),
    "dark_purple_alt": np.array([63, 0, 63], dtype=np.uint8),
    "dark_cyan": np.array([0, 63, 127], dtype=np.uint8),
    "khaki": np.array([127, 63, 0], dtype=np.uint8),
    "orange": np.array([255, 127, 0], dtype=np.uint8)
    
}

# 定义标签
labels = [
    'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle',
    'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface',
    'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation'
]

# 创建一个空白图像
height, width = 400, 1200
image = np.zeros((height, width, 3), dtype=np.uint8)

# 定义每个颜色块的大小
block_size = 30
text_size = 0.5
text_color = (255, 255, 255)

# 绘制颜色块和文本
for idx, (name, color) in enumerate(colors.items()):
    if idx < len(labels):
        label = labels[idx]
    else:
        label = "unknown"

    row = idx // 4
    col = idx % 4
    x_start = col * (block_size + 180)
    y_start = row * (block_size + 30)  # 增加一些间距以适应文本
    cv2.rectangle(image, (x_start, y_start), (x_start + block_size, y_start + block_size), color.tolist(), -1)
    cv2.putText(image, label, (x_start, y_start + block_size + 20), cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, 1)

# 显示图像
cv2.imshow('Colors', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
