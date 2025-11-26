import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import seaborn as sns
from tqdm import tqdm
import argparse
from pathlib import Path
import cv2

class MedicalDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 遍历数据目录，获取所有图像路径和标签
        for img_path in self.data_dir.glob('*.jpg'):
            if img_path.name.startswith('disease'):
                label = 1  # 患病
            elif img_path.name.startswith('normal'):
                label = 0  # 正常
            else:
                continue  # 跳过不符合命名规则的文件
            
            self.image_paths.append(img_path)
            self.labels.append(label)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class MedicalCNN(nn.Module):
    def __init__(self, pretrained=True):
        super(MedicalCNN, self).__init__()
        # 使用预训练的ResNet18作为基础模型
        self.base_model = models.resnet18(pretrained=pretrained)
        
        # 修改最后的全连接层以适应二分类任务
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        x = self.base_model(x)
        return x

class SimpleMedicalCNN(nn.Module):
    def __init__(self):
        super(SimpleMedicalCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 14 * 14, 512)  # 假设输入图像大小为224x224
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def calculate_class_weights(labels):
    """计算类别权重以处理类别不平衡"""
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return torch.FloatTensor(class_weights)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, class_weights=None):
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
            images, labels = images.to(device), labels.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
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
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float()
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(torch.sigmoid(outputs).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 计算AUC
        val_auc = roc_auc_score(all_labels, all_predictions)
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val AUC: {val_auc:.4f}')
    
    return train_losses, val_losses, train_accs, val_accs

def evaluate_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float()
            outputs = model(images).squeeze()
            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).float()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)

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
    plt.savefig('problem2/training_curves.png')
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Disease'], yticklabels=['Normal', 'Disease'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('problem2/confusion_matrix.png')
    plt.show()

def plot_roc_curve(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('problem2/roc_curve.png')
    plt.show()

def visualize_predictions(model, test_loader, device, num_samples=8):
    model.eval()
    images_shown = 0
    
    plt.figure(figsize=(16, 8))
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float()
            outputs = model(images).squeeze()
            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).float()
            
            for i in range(min(len(images), num_samples - images_shown)):
                plt.subplot(2, 4, images_shown + 1)
                
                # 反标准化图像以便显示
                img_display = images[i].cpu().permute(1, 2, 0).numpy()
                img_display = np.clip(img_display * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)
                
                plt.imshow(img_display)
                true_label = 'Disease' if labels[i].item() == 1 else 'Normal'
                pred_label = 'Disease' if predicted[i].item() == 1 else 'Normal'
                prob = probabilities[i].item()
                
                plt.title(f'True: {true_label}\nPred: {pred_label}\nProb: {prob:.3f}')
                plt.axis('off')
                images_shown += 1
                
                if images_shown >= num_samples:
                    break
            
            if images_shown >= num_samples:
                break
    
    plt.tight_layout()
    plt.savefig('problem2/predictions.png')
    plt.show()

def analyze_failure_cases(model, test_loader, device, dataset):
    """分析失败案例"""
    model.eval()
    failure_cases = []
    
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device).float()
            outputs = model(images).squeeze()
            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).float()
            
            for i in range(len(images)):
                if predicted[i] != labels[i]:
                    failure_cases.append({
                        'index': idx * test_loader.batch_size + i,
                        'true_label': labels[i].item(),
                        'predicted_label': predicted[i].item(),
                        'probability': probabilities[i].item(),
                        'image_path': dataset.image_paths[idx * test_loader.batch_size + i]
                    })
    
    return failure_cases

def main():
    parser = argparse.ArgumentParser(description='Medical Image Detection')
    parser.add_argument('--train_dir', type=str, default='2-MedImage-TrainSet', 
                       help='Path to training data directory')
    parser.add_argument('--test_dir', type=str, default='2-MedImage-TestSet', 
                       help='Path to test data directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--model_type', type=str, default='simple', choices=['simple', 'resnet'],
                       help='Model type: simple or resnet')
    
    args = parser.parse_args()
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 数据预处理和增强
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    try:
        full_train_dataset = MedicalDataset(args.train_dir, transform=train_transform)
        test_dataset = MedicalDataset(args.test_dir, transform=test_transform)
        
        # 划分训练集和验证集
        train_size = int(0.8 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_train_dataset, [train_size, val_size])
        
        print(f'Training samples: {len(train_dataset)}')
        print(f'Validation samples: {len(val_dataset)}')
        print(f'Test samples: {len(test_dataset)}')
        
        # 统计类别分布
        train_labels = [full_train_dataset.labels[i] for i in train_dataset.indices]
        from collections import Counter
        label_counts = Counter(train_labels)
        print(f"训练集类别分布: {label_counts}")
        
    except FileNotFoundError:
        print("数据目录未找到。请确保数据集已正确下载并解压。")
        print("训练数据目录: 2-MedImage-TrainSet")
        print("测试数据目录: 2-MedImage-TestSet")
        return
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 创建模型
    if args.model_type == 'resnet':
        model = MedicalCNN(pretrained=True).to(device)
    else:
        model = SimpleMedicalCNN().to(device)
    
    # 计算类别权重
    class_weights = calculate_class_weights(train_labels)
    pos_weight = class_weights[1] / class_weights[0]
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    # 训练模型
    print("开始训练...")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, args.num_epochs, device)
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    
    # 在测试集上评估
    print("在测试集上评估...")
    predictions, true_labels, probabilities = evaluate_model(model, test_loader, device)
    
    # 计算各种指标
    accuracy = accuracy_score(true_labels, predictions)
    auc = roc_auc_score(true_labels, probabilities)
    
    print(f'测试集准确率: {accuracy:.4f}')
    print(f'测试集AUC: {auc:.4f}')
    
    # 打印详细的分类报告
    print("\n分类报告:")
    print(classification_report(true_labels, predictions, target_names=['Normal', 'Disease'], digits=4))
    
    # 绘制混淆矩阵
    plot_confusion_matrix(true_labels, predictions)
    
    # 绘制ROC曲线
    plot_roc_curve(true_labels, probabilities)
    
    # 可视化预测结果
    visualize_predictions(model, test_loader, device)
    
    # 分析失败案例
    print("\n分析失败案例...")
    failure_cases = analyze_failure_cases(model, test_loader, device, test_dataset)
    print(f"失败案例数量: {len(failure_cases)}")
    
    # 显示一些失败案例
    if failure_cases:
        print("\n失败案例示例:")
        for i, case in enumerate(failure_cases[:5]):
            true_label = 'Disease' if case['true_label'] == 1 else 'Normal'
            pred_label = 'Disease' if case['predicted_label'] == 1 else 'Normal'
            print(f"案例 {i+1}: 真实={true_label}, 预测={pred_label}, 概率={case['probability']:.3f}")
    
    # 保存模型
    torch.save(model.state_dict(), f'problem2/medical_model_{args.model_type}.pth')
    print(f"模型已保存到 problem2/medical_model_{args.model_type}.pth")

if __name__ == '__main__':
    main()