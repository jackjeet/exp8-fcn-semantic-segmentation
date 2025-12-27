import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random


# 自定义模拟语义分割数据集（替代Pascal VOC，跳过下载）
class FakeSegDataset(Dataset):
    def __init__(self, num_samples=100, img_size=(256, 256), num_classes=21):
        self.num_samples = num_samples  # 模拟100张训练图
        self.img_size = img_size  # 统一尺寸
        self.num_classes = num_classes  # 21类（适配FCN标准）
        # 图像预处理（和真实数据集一致）
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 生成模拟图像（3通道，256x256）
        fake_img = np.random.randint(0, 255, size=(*self.img_size, 3), dtype=np.uint8)
        # 生成模拟像素级标签（单通道，每个像素值对应类别）
        fake_label = np.random.randint(0, self.num_classes, size=self.img_size, dtype=np.uint8)

        # 预处理
        img = self.transform(fake_img)
        label = torch.from_numpy(fake_label).long()  # 标签转为long型
        return img, label


# 生成数据集并验证
if __name__ == "__main__":
    # 创建模拟训练集
    train_dataset = FakeSegDataset(num_samples=100)
    # 创建数据加载器（适配模型训练）
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # 验证数据集可加载
    imgs, labels = next(iter(train_loader))
    print(f"✅ 模拟语义分割数据集创建成功！")
    print(f"批量图像尺寸：{imgs.shape}（batch=4, 通道=3, 高=256, 宽=256）")
    print(f"批量标签尺寸：{labels.shape}（batch=4, 高=256, 宽=256）")
    print(f"标签类别范围：{labels.min()} ~ {labels.max()}")