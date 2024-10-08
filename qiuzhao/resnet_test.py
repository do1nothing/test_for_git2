import torch
import torchvision
from torchsummary import summary

# 检查是否有可用的GPU
# if torch.cuda.is_available():
#     device = torch.device("cuda")  # 指定设备为GPU
# else:
device = torch.device("cpu")  # 否则使用CPU

# 加载模型并将其移动到指定设备
model = torchvision.models.resnet18().to(device)

# 打印模型摘要
summary(model, (3, 224, 224), device=device)
# x = torch.randn(4,3,224,224)
# import pdb;pdb.set_trace()
# y = model(x)
