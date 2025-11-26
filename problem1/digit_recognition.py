import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import argparse
from pathlib import Path

class DigitDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 遍历数据目录，获取所有图像路径和标签
        for img_path in self.data_dir.glob('*.bmp'):
            # 文件名的第一个数字是标签
            label = int(img_path.name[0])
            self.image_paths.append(img_path)
            self.labels.append(label)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 加载图像
        image = Image.open(img_path).convert('L')  # 转换为灰度图
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)  # 假设输入图像大小为28x28
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    return train_losses, val_losses, train_accs, val_accs

def evaluate_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels)

def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # 准确率曲线
    ax2.plot(train_accs, label='Training Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('problem1/training_curves.png')
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('problem1/confusion_matrix.png')
    plt.show()

def visualize_predictions(model, test_loader, device, num_samples=10):
    model.eval()
    images_shown = 0
    
    plt.figure(figsize=(15, 8))
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            for i in range(min(len(images), num_samples - images_shown)):
                plt.subplot(2, 5, images_shown + 1)
                plt.imshow(images[i].cpu().squeeze(), cmap='gray')
                plt.title(f'True: {labels[i].item()}, Pred: {predicted[i].item()}')
                plt.axis('off')
                images_shown += 1
                
                if images_shown >= num_samples:
                    break
            
            if images_shown >= num_samples:
                break
    
    plt.tight_layout()
    plt.savefig('problem1/predictions.png')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Handwritten Digit Recognition')
    parser.add_argument('--train_dir', type=str, default='1-Digit-TrainSet', 
                       help='Path to training data directory')
    parser.add_argument('--test_dir', type=str, default='1-Digit-TestSet', 
                       help='Path to test data directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准化参数
    ])
    
    # 创建数据集
    try:
        full_train_dataset = DigitDataset(args.train_dir, transform=transform)
        test_dataset = DigitDataset(args.test_dir, transform=transform)
        
        # 划分训练集和验证集
        train_size = int(0.8 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_train_dataset, [train_size, val_size])
        
        print(f'Training samples: {len(train_dataset)}')
        print(f'Validation samples: {len(val_dataset)}')
        print(f'Test samples: {len(test_dataset)}')
        
    except FileNotFoundError:
        print("数据目录未找到。请确保数据集已正确下载并解压。")
        print("训练数据目录: 1-Digit-TrainSet")
        print("测试数据目录: 1-Digit-TestSet")
        return
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 创建模型
    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 训练模型
    print("开始训练...")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, args.num_epochs, device)
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    
    # 在测试集上评估
    print("在测试集上评估...")
    predictions, true_labels = evaluate_model(model, test_loader, device)
    
    # 计算准确率
    accuracy = accuracy_score(true_labels, predictions)
    print(f'测试集准确率: {accuracy:.4f}')
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(true_labels, predictions, digits=4))
    
    # 绘制混淆矩阵
    plot_confusion_matrix(true_labels, predictions)
    
    # 可视化预测结果
    visualize_predictions(model, test_loader, device)
    
    # 保存模型
    torch.save(model.state_dict(), 'problem1/digit_model.pth')
    print("模型已保存到 problem1/digit_model.pth")

if __name__ == '__main__':
    main()