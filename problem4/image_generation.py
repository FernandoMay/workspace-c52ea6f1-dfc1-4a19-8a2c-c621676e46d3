import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from pathlib import Path
import torch_fidelity
import random

# 设置随机种子以确保结果可重现
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 输入: nz x 1 x 1
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 状态: (ngf*8) x 4 x 4
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 状态: (ngf*4) x 8 x 8
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 状态: (ngf*2) x 16 x 16
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 状态: (ngf) x 32 x 32
            
            nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False),
            nn.Tanh()
            # 输出: nc x 32 x 32
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 输入: nc x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态: (ndf) x 16 x 16
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态: (ndf*2) x 8 x 8
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态: (ndf*4) x 4 x 4
            
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

def weights_init(m):
    """自定义权重初始化"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def load_cifar10_data(batch_size=64):
    """加载CIFAR-10数据集"""
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # 下载数据集
    train_dataset = datasets.CIFAR10(root='./CIFARdata', train=True, 
                                   download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./CIFARdata', train=False, 
                                  download=True, transform=transform)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=2)
    
    return train_loader, test_loader

def save_generated_images(generator, device, epoch, num_images=64):
    """保存生成的图像"""
    generator.eval()
    with torch.no_grad():
        # 生成随机噪声
        noise = torch.randn(num_images, 100, 1, 1, device=device)
        fake_images = generator(noise)
        
        # 保存图像网格
        vutils.save_image(fake_images, 
                        f'problem4/generated_images_epoch_{epoch}.png',
                        nrow=8, normalize=True)
    
    generator.train()

def plot_training_progress(g_losses, d_losses):
    """绘制训练进度"""
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('problem4/training_progress.png')
    plt.show()

def visualize_real_vs_fake(real_dataloader, generator, device, num_samples=8):
    """可视化真实图像 vs 生成图像"""
    generator.eval()
    
    # 获取真实图像
    real_batch = next(iter(real_dataloader))
    real_images = real_batch[0][:num_samples]
    
    # 生成假图像
    with torch.no_grad():
        noise = torch.randn(num_samples, 100, 1, 1, device=device)
        fake_images = generator(noise)
    
    # 反归一化
    def denorm(img):
        return img * 0.5 + 0.5
    
    real_images = denorm(real_images)
    fake_images = denorm(fake_images)
    
    # 绘制图像
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i in range(num_samples):
        # 真实图像
        axes[0, i].imshow(real_images[i].permute(1, 2, 0))
        axes[0, i].set_title('Real')
        axes[0, i].axis('off')
        
        # 生成图像
        axes[1, i].imshow(fake_images[i].cpu().permute(1, 2, 0))
        axes[1, i].set_title('Generated')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('problem4/real_vs_fake.png')
    plt.show()
    
    generator.train()

def calculate_fidelity_metrics(generator, real_dataloader, device, num_samples=1000):
    """计算FID, IS, KID等指标"""
    generator.eval()
    
    # 创建临时目录保存图像
    os.makedirs('temp_generated', exist_ok=True)
    os.makedirs('temp_real', exist_ok=True)
    
    # 保存真实图像
    real_count = 0
    for batch in real_dataloader:
        real_images = batch[0]
        for i in range(real_images.size(0)):
            if real_count >= num_samples:
                break
            # 反归一化并保存
            img = real_images[i] * 0.5 + 0.5
            vutils.save_image(img, f'temp_real/real_{real_count:04d}.png')
            real_count += 1
        if real_count >= num_samples:
            break
    
    # 生成并保存假图像
    with torch.no_grad():
        for i in range(0, num_samples, 64):
            batch_size = min(64, num_samples - i)
            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            fake_images = generator(noise)
            
            for j in range(batch_size):
                img = fake_images[j] * 0.5 + 0.5
                vutils.save_image(img, f'temp_generated/fake_{i+j:04d}.png')
    
    # 计算指标
    try:
        metrics_dict = torch_fidelity.calculate_metrics(
            input1='temp_generated',
            input2='temp_real',
            cuda=True,
            isc=True,
            fid=True,
            kid=True,
            verbose=False
        )
        
        print("Fidelity Metrics:")
        print(f"  Inception Score (IS): {metrics_dict.get('isc', 'N/A')}")
        print(f"  Frechet Inception Distance (FID): {metrics_dict.get('fid', 'N/A')}")
        print(f"  Kernel Inception Distance (KID): {metrics_dict.get('kid', 'N/A')}")
        
        return metrics_dict
        
    except Exception as e:
        print(f"Error calculating fidelity metrics: {e}")
        return {}
    
    finally:
        # 清理临时文件
        import shutil
        shutil.rmtree('temp_generated', ignore_errors=True)
        shutil.rmtree('temp_real', ignore_errors=True)
    
    generator.train()

def train_gan(dataloader, generator, discriminator, device, num_epochs, nz):
    """训练GAN"""
    # 损失函数
    criterion = nn.BCELoss()
    
    # 优化器
    optimizerG = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerD = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # 记录损失
    G_losses = []
    D_losses = []
    
    # 固定噪声用于可视化
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    
    print("开始训练GAN...")
    for epoch in range(num_epochs):
        for i, data in enumerate(tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            ############################
            # (1) 更新判别器D
            ###########################
            # 训练真实图像
            discriminator.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), 1, dtype=torch.float, device=device)
            
            output = discriminator(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            
            # 训练生成图像
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = generator(noise)
            label.fill_(0)
            output = discriminator(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()
            
            ############################
            # (2) 更新生成器G
            ###########################
            generator.zero_grad()
            label.fill_(1)
            output = discriminator(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            
            # 记录损失
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
            # 打印进度
            if i % 50 == 0:
                print(f'[{epoch+1}/{num_epochs}][{i}/{len(dataloader)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                      f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')
        
        # 保存生成的图像
        if epoch % 5 == 0:
            save_generated_images(generator, device, epoch)
    
    return G_losses, D_losses

def generate_class_specific_samples(generator, device, num_classes=10, samples_per_class=8):
    """生成特定类别的样本（这里只是随机生成，因为GAN是无条件的）"""
    generator.eval()
    
    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(samples_per_class*2, num_classes*2))
    
    for class_idx in range(num_classes):
        with torch.no_grad():
            noise = torch.randn(samples_per_class, 100, 1, 1, device=device)
            generated = generator(noise)
            
            for i in range(samples_per_class):
                img = generated[i] * 0.5 + 0.5
                axes[class_idx, i].imshow(img.cpu().permute(1, 2, 0))
                axes[class_idx, i].axis('off')
                
                if i == 0:
                    axes[class_idx, i].set_ylabel(f'Class {class_idx}', fontsize=10)
    
    plt.suptitle('Generated Samples by Class (Random)', fontsize=14)
    plt.tight_layout()
    plt.savefig('problem4/class_specific_samples.png')
    plt.show()
    
    generator.train()

def main():
    parser = argparse.ArgumentParser(description='Color Image Generation with GAN')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--nz', type=int, default=100, help='Size of z latent vector')
    parser.add_argument('--ngf', type=int, default=64, help='Generator feature maps')
    parser.add_argument('--ndf', type=int, default=64, help='Discriminator feature maps')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for Adam optimizer')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 创建模型
    generator = Generator(args.nz, args.ngf).to(device)
    discriminator = Discriminator().to(device)
    
    # 初始化权重
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # 加载数据
    print("加载CIFAR-10数据集...")
    train_loader, test_loader = load_cifar10_data(args.batch_size)
    
    print(f"训练样本数: {len(train_loader.dataset)}")
    print(f"测试样本数: {len(test_loader.dataset)}")
    
    # 训练GAN
    G_losses, D_losses = train_gan(train_loader, generator, discriminator, 
                                  device, args.num_epochs, args.nz)
    
    # 绘制训练进度
    plot_training_progress(G_losses, D_losses)
    
    # 可视化真实图像 vs 生成图像
    visualize_real_vs_fake(test_loader, generator, device)
    
    # 生成类别特定样本
    generate_class_specific_samples(generator, device)
    
    # 计算Fidelity指标
    print("计算Fidelity指标...")
    metrics = calculate_fidelity_metrics(generator, test_loader, device)
    
    # 保存最终模型
    torch.save({
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'args': args
    }, 'problem4/gan_model.pth')
    
    print("模型已保存到 problem4/gan_model.pth")
    print("训练完成！")

if __name__ == '__main__':
    main()