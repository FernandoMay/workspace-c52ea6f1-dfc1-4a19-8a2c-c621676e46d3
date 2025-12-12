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
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mean_squared_error
import seaborn as sns
from tqdm import tqdm
import argparse
from pathlib import Path
import cv2

class SaliencyDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.image_paths = []
        self.saliency_paths = []
        
        # 遍历Stimuli文件夹获取图像路径
        stimuli_dir = self.data_dir / 'Stimuli'
        fixation_dir = self.data_dir / 'FIXATIONMAPS'
        
        if not stimuli_dir.exists() or not fixation_dir.exists():
            raise FileNotFoundError(f"Stimuli或FIXATIONMAPS文件夹不存在在 {data_dir}")
        
        # 递归遍历所有子文件夹
        for img_path in stimuli_dir.rglob('*.jpg'):
            # 找到对应的显著图
            relative_path = img_path.relative_to(stimuli_dir)
            saliency_path = fixation_dir / relative_path
            
            if saliency_path.exists():
                self.image_paths.append(img_path)
                self.saliency_paths.append(saliency_path)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        saliency_path = self.saliency_paths[idx]
        
        # 加载原始图像
        image = Image.open(img_path).convert('RGB')
        
        # 加载显著图（灰度图）
        saliency = Image.open(saliency_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            saliency = self.target_transform(saliency)
        
        return image, saliency

class SaliencyCNN(nn.Module):
    def __init__(self):
        super(SaliencyCNN, self).__init__()
        
        # 编码器部分
        self.encoder = nn.Sequential(
            # 输入: 3 x 256 x 256
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 64 x 256 x 256
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 64 x 128 x 128
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 128 x 128 x 128
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 128 x 64 x 64
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 256 x 64 x 64
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 256 x 32 x 32
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 512 x 32 x 32
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 512 x 16 x 16
        )
        
        # 解码器部分
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 512 x 32 x 32
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 512 x 32 x 32
            
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 256 x 64 x 64
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 256 x 64 x 64
            
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 128 x 128 x 128
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 128 x 128 x 128
            
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 64 x 256 x 256
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 64 x 256 x 256
            
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
            # 1 x 256 x 256
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class SaliencyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(SaliencyLoss, self).__init__()
        self.alpha = alpha  # MSE权重
        self.beta = beta    # CC权重
        self.mse_loss = nn.MSELoss()
        
    def correlation_coefficient(self, pred, target):
        """计算相关系数"""
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # 计算均值
        pred_mean = torch.mean(pred_flat)
        target_mean = torch.mean(target_flat)
        
        # 计算协方差和方差
        pred_var = torch.var(pred_flat)
        target_var = torch.var(target_flat)
        
        # 计算相关系数
        cov = torch.mean((pred_flat - pred_mean) * (target_flat - target_mean))
        cc = cov / (torch.sqrt(pred_var) * torch.sqrt(target_var) + 1e-8)
        
        return cc
    
    def forward(self, pred, target):
        mse = self.mse_loss(pred, target)
        cc = self.correlation_coefficient(pred, target)
        
        # 总损失 = MSE - CC (因为我们要最大化CC)
        total_loss = self.alpha * mse - self.beta * cc
        
        return total_loss, mse, cc

def calculate_metrics(pred, target):
    """计算各种评价指标"""
    # 转换为numpy数组
    pred_np = pred.cpu().numpy().flatten()
    target_np = target.cpu().numpy().flatten()
    
    # 归一化
    pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)
    target_np = (target_np - target_np.min()) / (target_np.max() - target_np.min() + 1e-8)
    
    # 计算指标
    mse = mean_squared_error(target_np, pred_np)
    cc, _ = pearsonr(target_np, pred_np)
    
    # 计算KL散度（对零值做掩码并平滑，避免log(0)或0 * inf导致NaN）
    eps = 1e-8
    pred_prob = pred_np / (pred_np.sum() + eps)
    target_prob = target_np / (target_np.sum() + eps)

    # 平滑并裁剪以避免除零或log(0)
    pred_prob = np.clip(pred_prob, eps, 1.0)
    target_prob = np.clip(target_prob, 0.0, 1.0)

    # 仅在target_prob>0的位置计算KL，避免0 * log(0/.)产生NaN
    mask = target_prob > 0
    if np.any(mask):
        kl_div = np.sum(target_prob[mask] * np.log(target_prob[mask] / pred_prob[mask]))
    else:
        kl_div = 0.0
    
    # 计算Jensen-Shannon散度
    # Jensen-Shannon散度（使用平滑后的分布）
    js_div = jensenshannon(pred_prob, target_prob) ** 2
    
    return {
        'mse': mse,
        'cc': cc if not np.isnan(cc) else 0.0,
        'kl_div': kl_div,
        'js_div': js_div
    }

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_mse = 0.0
        running_cc = 0.0
        num_batches = 0
        
        for images, saliencies in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images, saliencies = images.to(device), saliencies.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss, mse, cc = criterion(outputs, saliencies)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_mse += mse.item()
            running_cc += cc.item()
            num_batches += 1
        
        train_loss = running_loss / num_batches
        train_mse = running_mse / num_batches
        train_cc = running_cc / num_batches
        train_losses.append(train_loss)
        train_metrics.append({'mse': train_mse, 'cc': train_cc})
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_cc = 0.0
        val_num_batches = 0
        
        with torch.no_grad():
            for images, saliencies in val_loader:
                images, saliencies = images.to(device), saliencies.to(device)
                outputs = model(images)
                loss, mse, cc = criterion(outputs, saliencies)
                
                val_loss += loss.item()
                val_mse += mse.item()
                val_cc += cc.item()
                val_num_batches += 1
        
        val_loss = val_loss / val_num_batches
        val_mse = val_mse / val_num_batches
        val_cc = val_cc / val_num_batches
        val_losses.append(val_loss)
        val_metrics.append({'mse': val_mse, 'cc': val_cc})
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train MSE: {train_mse:.4f}, Train CC: {train_cc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val MSE: {val_mse:.4f}, Val CC: {val_cc:.4f}')
    
    return train_losses, val_losses, train_metrics, val_metrics

def evaluate_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_targets = []
    all_metrics = []
    
    with torch.no_grad():
        for images, saliencies in tqdm(test_loader, desc='Evaluating'):
            images, saliencies = images.to(device), saliencies.to(device)
            outputs = model(images)
            
            # 计算指标
            metrics = calculate_metrics(outputs, saliencies)
            all_metrics.append(metrics)
            
            all_predictions.append(outputs.cpu())
            all_targets.append(saliencies.cpu())
    
    # 计算平均指标（对可能出现的NaN使用nanmean）
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.nanmean([m[key] for m in all_metrics])
    
    return all_predictions, all_targets, avg_metrics

def plot_training_curves(train_losses, val_losses, train_metrics, val_metrics):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 损失曲线
    axes[0, 0].plot(train_losses, label='Training Loss')
    axes[0, 0].plot(val_losses, label='Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    
    # MSE曲线
    train_mse = [m['mse'] for m in train_metrics]
    val_mse = [m['mse'] for m in val_metrics]
    axes[0, 1].plot(train_mse, label='Training MSE')
    axes[0, 1].plot(val_mse, label='Validation MSE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].set_title('Training and Validation MSE')
    axes[0, 1].legend()
    
    # CC曲线
    train_cc = [m['cc'] for m in train_metrics]
    val_cc = [m['cc'] for m in val_metrics]
    axes[1, 0].plot(train_cc, label='Training CC')
    axes[1, 0].plot(val_cc, label='Validation CC')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Correlation Coefficient')
    axes[1, 0].set_title('Training and Validation CC')
    axes[1, 0].legend()
    
    # 指标对比
    final_metrics = ['MSE', 'CC']
    final_train = [train_mse[-1], train_cc[-1]]
    final_val = [val_mse[-1], val_cc[-1]]
    
    x = np.arange(len(final_metrics))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, final_train, width, label='Training')
    axes[1, 1].bar(x + width/2, final_val, width, label='Validation')
    axes[1, 1].set_xlabel('Metrics')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Final Metrics Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(final_metrics)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('problem3/training_curves.png')
    plt.show()

def visualize_predictions(predictions, targets, num_samples=4):
    """可视化预测结果"""
    num_samples = min(num_samples, len(predictions))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for i in range(num_samples):
        pred = predictions[i][0].numpy()
        target = targets[i][0].numpy()

        # 去掉通道轴 (1, H, W) -> (H, W)
        pred = np.squeeze(pred)
        target = np.squeeze(target)
        
        # 归一化
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        target = (target - target.min()) / (target.max() - target.min() + 1e-8)
        
        # 原始图像（这里只显示显著图）
        axes[i, 0].imshow(target, cmap='gray')
        axes[i, 0].set_title('Ground Truth')
        axes[i, 0].axis('off')
        
        # 预测结果
        axes[i, 1].imshow(pred, cmap='gray')
        axes[i, 1].set_title('Prediction')
        axes[i, 1].axis('off')
        
        # 差异图
        diff = np.abs(pred - target)
        axes[i, 2].imshow(diff, cmap='hot')
        axes[i, 2].set_title('Difference')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('problem3/predictions.png')
    plt.show()

def analyze_by_category(test_loader, model, device, dataset):
    """按类别分析结果"""
    # 这里需要根据实际的数据集结构来实现
    # 暂时返回空结果
    return {}

def main():
    parser = argparse.ArgumentParser(description='Image Saliency Prediction')
    parser.add_argument('--train_dir', type=str, default='3-Saliency-TrainSet', 
                       help='Path to training data directory')
    parser.add_argument('--test_dir', type=str, default='3-Saliency-TestSet', 
                       help='Path to test data directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    target_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # 创建数据集
    try:
        full_train_dataset = SaliencyDataset(args.train_dir, transform=transform, 
                                           target_transform=target_transform)
        test_dataset = SaliencyDataset(args.test_dir, transform=transform, 
                                     target_transform=target_transform)
        
        # 划分训练集和验证集
        train_size = int(0.8 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_train_dataset, [train_size, val_size])
        
        print(f'Training samples: {len(train_dataset)}')
        print(f'Validation samples: {len(val_dataset)}')
        print(f'Test samples: {len(test_dataset)}')
        
    except FileNotFoundError as e:
        print(f"数据目录未找到: {e}")
        print("请确保数据集已正确下载并解压，包含Stimuli和FIXATIONMAPS文件夹")
        print("训练数据目录: 3-Saliency-TrainSet")
        print("测试数据目录: 3-Saliency-TestSet")
        return
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 创建模型
    model = SaliencyCNN().to(device)
    criterion = SaliencyLoss(alpha=0.5, beta=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    # 训练模型
    print("开始训练...")
    train_losses, val_losses, train_metrics, val_metrics = train_model(
        model, train_loader, val_loader, criterion, optimizer, args.num_epochs, device)
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, train_metrics, val_metrics)
    
    # 在测试集上评估
    print("在测试集上评估...")
    predictions, targets, avg_metrics = evaluate_model(model, test_loader, device)
    
    print(f"测试集平均指标:")
    for metric, value in avg_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 可视化预测结果
    visualize_predictions(predictions, targets)
    
    # 按类别分析
    category_results = analyze_by_category(test_loader, model, device, test_dataset)
    
    # 保存模型
    torch.save(model.state_dict(), 'problem3/saliency_model.pth')
    print("模型已保存到 problem3/saliency_model.pth")

if __name__ == '__main__':
    main()