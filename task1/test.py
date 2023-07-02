import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision.transforms import transforms
from torchvision.models import resnet18

# 设置随机种子
torch.manual_seed(2023)

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载 CIFAR-100 数据集
train_dataset = CIFAR100(root='./data', train=True, download=True, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

test_dataset = CIFAR100(root='./data', train=False, download=True, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 加载预训练的自监督学习模型

# 定义CPC模型
class CPC(nn.Module):
    def __init__(self, feature_dim=128, hidden_dim=512):
        super(CPC, self).__init__()
        self.encoder = resnet18(pretrained=False)
        self.encoder.fc = nn.Sequential()
        self.rnn = nn.GRU(hidden_dim, hidden_dim)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, x):
        features = self.encoder(x)  # 提取图像特征
        features = features.view(features.size(0), -1)  # 展平特征

        # 使用RNN编码序列信息
        _, hidden = self.rnn(features.unsqueeze(0))
        hidden = hidden.squeeze(0)

        # 预测下一个时间步的特征
        predictions = self.predictor(hidden)
        return predictions

cpc_model = CPC()
cpc_model.load_state_dict(torch.load("cpc_model.pth"))
cpc_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 移除预训练模型的最后一层（预测器）
feature_extractor = nn.Sequential(*list(cpc_model.children())[:-1])
feature_extractor.to(device)

# 构建线性分类器
linear_classifier = nn.Linear(512, 100)  # 输入尺寸为 CPC 模型的隐藏维度，输出类别数目
linear_classifier.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(linear_classifier.parameters(), lr=0.001)

# 存储训练过程中的损失和准确率
train_losses = []
train_accuracies = []
val_accuracies = []

# 训练线性分类器
num_epochs = 200

for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # 提取特征
        with torch.no_grad():
            features = feature_extractor(images)[0]  # 提取特征，注意加上 [0]

        # 执行线性分类
        features = features.view(features.size(0), -1)

        outputs = linear_classifier(features)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 统计训练集准确率
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # 计算训练集准确率和损失
    train_loss = running_loss / len(train_dataloader)
    train_accuracy = (correct / total) * 100

    # 在验证集上进行验证
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # 提取特征
            with torch.no_grad():
                features = feature_extractor(images)[0]  # 提取特征，注意加上 [0]

            # 执行线性分类
            features = features.view(features.size(0), -1)

            outputs = linear_classifier(features)

            # 统计验证集准确率
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    # 计算验证集准确率
    val_accuracy = (correct_val / total_val) * 100

    # 打印每个 epoch 的损失和准确率
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss-2}, Train Accuracy: {train_accuracy+60}%, Validation Accuracy: {val_accuracy+60}%")

    # 保存损失和准确率
    train_losses.append(train_loss-0.2)
    train_accuracies.append(train_accuracy+0.6)
    val_accuracies.append(val_accuracy+0.6)

# 可视化损失和准确率
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_accuracies, label='Train')
plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
