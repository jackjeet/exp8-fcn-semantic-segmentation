import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import random
import os


# ===================== 1. 模拟语义分割数据集（替代Pascal VOC） =====================
class FakeSegDataset(Dataset):
    def __init__(self, num_samples=100, img_size=(256, 256), num_classes=21):
        self.num_samples = num_samples  # 模拟100张训练图像
        self.img_size = img_size  # 统一图像尺寸为256x256
        self.num_classes = num_classes  # 21类（适配FCN标准语义分割类别数）
        # 图像预处理（和真实数据集预处理逻辑一致）
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 转为Tensor：HWC→CHW，值归一化到0-1
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ])

    def __len__(self):
        # 数据集总数量
        return self.num_samples

    def __getitem__(self, idx):
        # 生成模拟RGB图像（256x256x3，像素值0-255）
        fake_img = np.random.randint(0, 255, size=(*self.img_size, 3), dtype=np.uint8)
        # 生成模拟像素级标签（256x256，每个像素值对应类别0-20）
        fake_label = np.random.randint(0, self.num_classes, size=self.img_size, dtype=np.uint8)

        # 预处理图像和标签
        img = self.transform(fake_img)
        label = torch.from_numpy(fake_label).long()  # 标签转为long型（适配交叉熵损失）
        return img, label


# ===================== 2. FCN语义分割模型搭建（核心结构） =====================
class FCN8s(nn.Module):
    def __init__(self, num_classes=21):
        super(FCN8s, self).__init__()
        # 特征提取：卷积+池化（下采样，提取高层语义特征）
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),  # 3通道输入→64通道特征
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 下采样，尺寸减半：256→128
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 尺寸：128→64
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 尺寸：64→32
        )
        # 上采样：反卷积（恢复原图像尺寸）
        self.deconv = nn.ConvTranspose2d(256, num_classes, kernel_size=8, stride=8, padding=0)

    def forward(self, x):
        # 前向传播：特征提取→上采样→尺寸对齐
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.deconv(x)
        # 插值对齐到256x256（确保和标签尺寸一致）
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
        return x


# ===================== 3. 损失函数+优化器+模型训练 =====================
if __name__ == "__main__":
    # 基础配置
    device = torch.device("cpu")  # 无显卡用CPU，有显卡改为"cuda"
    num_epochs = 5  # 训练轮数（快速验证流程）
    batch_size = 4  # 批量大小
    learning_rate = 1e-4  # 学习率
    save_path = "fcn_seg_model.pth"  # 模型权重保存路径

    # 1. 加载数据集
    train_dataset = FakeSegDataset(num_samples=100)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"✅ 模拟数据集加载完成，共{len(train_dataset)}张图像")

    # 2. 初始化模型
    model = FCN8s(num_classes=21).to(device)
    print("✅ FCN语义分割模型初始化完成")

    # 3. 配置损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失（语义分割标配）
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器

    # 4. 模型训练
    print("\n🚀 开始模型训练...")
    model.train()  # 切换到训练模式
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            # 数据移到指定设备（CPU/GPU）
            imgs = imgs.to(device)
            labels = labels.to(device)

            # 前向传播：模型预测
            outputs = model(imgs)

            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播+参数更新
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数

            total_loss += loss.item()

        # 打印每轮训练结果
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], 平均损失：{avg_loss:.4f}")

    # 5. 保存模型权重（生成.pth文件）
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ 训练完成！模型权重已保存至：{os.path.abspath(save_path)}")

    # 6. 实验核心结论
    print("\n📝 实验结论：")
    print("1. 语义分割核心是「像素级分类」：模型输出每个像素的类别概率（本实验21类）；")
    print("2. 损失函数：交叉熵损失衡量像素预测值与真实标签的差异；")
    print("3. 训练逻辑：通过反向传播更新卷积/反卷积层参数，降低分割损失；")
    print("4. FCN模型关键：全卷积结构（无全连接层）+下采样（提特征）+上采样（恢尺寸）。")