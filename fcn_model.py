import torch
import torch.nn as nn
import torch.nn.functional as F


# 最简版FCN模型（保留语义分割核心逻辑：全卷积+上采样）
class FCN8s(nn.Module):
    def __init__(self, num_classes=21):
        super(FCN8s, self).__init__()
        # 第一步：卷积+池化（特征提取，模拟VGG主干）
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),  # 3通道输入→64通道特征
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 下采样，尺寸减半
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        # 第二步：反卷积（上采样，恢复原图像尺寸）
        self.deconv = nn.ConvTranspose2d(256, num_classes, kernel_size=8, stride=8, padding=0)

    # 前向传播：图像→特征提取→上采样→像素级分类
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.deconv(x)
        # 调整输出尺寸（确保和标签匹配）
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
        return x


# 验证模型搭建+适配模拟数据集
if __name__ == "__main__":
    # 1. 初始化FCN模型
    model = FCN8s(num_classes=21)
    print("✅ FCN语义分割模型搭建成功！")

    # 2. 加载之前的模拟数据集
    from fake_seg_dataset import FakeSegDataset

    train_dataset = FakeSegDataset(num_samples=100)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4)

    # 3. 验证模型可处理数据（前向传播）
    imgs, labels = next(iter(train_loader))
    output = model(imgs)

    print(f"模型输入尺寸（批量图像）：{imgs.shape}")
    print(f"模型输出尺寸（像素级分类）：{output.shape}")
    print(f"输出维度说明：[batch=4, 类别数=21, 高=256, 宽=256] → 每个像素对应21类的预测概率")