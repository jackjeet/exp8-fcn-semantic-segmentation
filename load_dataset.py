import torch
from torchvision import datasets, transforms
from torchvision.datasets import VOCSegmentation
import os

# 数据集保存路径（自动创建）
data_path = "./voc_seg_data"
os.makedirs(data_path, exist_ok=True)

# 预处理：图像归一化+尺寸统一（适配FCN模型）
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 统一尺寸
    transforms.ToTensor(),  # 转为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 标签预处理（仅尺寸统一）
target_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# 自动下载Pascal VOC 2012语义分割数据集（约2GB，耐心等）
train_dataset = VOCSegmentation(
    root=data_path,
    year='2012',
    image_set='train',
    download=True,
    transform=transform,
    target_transform=target_transform
)

# 验证数据集加载成功
img, label = train_dataset[0]
print(f"✅ 数据集下载成功！")
print(f"单张图像尺寸：{img.shape}")
print(f"单张标签尺寸：{label.shape}")
print(f"训练集总数：{len(train_dataset)}")