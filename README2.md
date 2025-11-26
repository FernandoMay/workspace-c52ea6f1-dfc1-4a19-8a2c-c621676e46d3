# ML-Class-Assignment25: 2025机器学习基础课程大作业

## 项目概述

本项目包含了2025年机器学习基础课程的四个大作业题目：
1. 手写数字识别
2. 医学图像检测
3. 图像显著性预测
4. 彩色图像生成

每个题目都提供了完整的Python实现代码、LaTeX报告模板（中文和西班牙语版本）以及详细的使用说明。

## 项目结构

```
ML-Class-Assignment25/
├── README.md                           # 项目说明文档
├── requirements.txt                     # Python依赖包列表
├── problem1/                           # 题目一：手写数字识别
│   └── digit_recognition.py            # 主要实现代码
├── problem2/                           # 题目二：医学图像检测
│   └── medical_detection.py            # 主要实现代码
├── problem3/                           # 题目三：图像显著性预测
│   └── saliency_prediction.py         # 主要实现代码
├── problem4/                           # 题目四：彩色图像生成
│   └── image_generation.py             # 主要实现代码
├── templates/                          # LaTeX报告模板
│   ├── report_template_chinese.tex             # 中文报告模板（题目一）
│   ├── report_template_chinese_problem2.tex     # 中文报告模板（题目二）
│   ├── report_template_chinese_problem3.tex     # 中文报告模板（题目三）
│   ├── report_template_chinese_problem4.tex     # 中文报告模板（题目四）
│   ├── report_template_spanish.tex             # 西班牙语报告模板（题目一）
│   ├── report_template_spanish_problem2.tex     # 西班牙语报告模板（题目二）
│   ├── report_template_spanish_problem3.tex     # 西班牙语报告模板（题目三）
│   └── report_template_spanish_problem4.tex     # 西班牙语报告模板（题目四）
└── utils/                              # 工具函数（可选）
```

## 环境要求

### 系统要求
- Python 3.8+
- CUDA支持的GPU（推荐，用于加速训练）
- 8GB+ RAM
- 10GB+ 可用磁盘空间

### 依赖包安装

```bash
# 创建虚拟环境（推荐）
conda create -n ml-assignment python=3.8
conda activate ml-assignment

# 安装依赖包
pip install -r requirements.txt
```

或者手动安装主要依赖：

```bash
pip install torch torchvision numpy matplotlib scikit-learn
pip install opencv-python pillow tqdm seaborn pandas
pip install tensorboard torch-fidelity scipy plotly
pip install jupyter ipywidgets
```

## 数据准备

### 数据下载链接

**百度云盘下载**：https://pan.baidu.com/s/1mOCFxATcCkHGbK8Vdtv5yQ

**DropBox下载**：https://www.dropbox.com/sh/i79cbllw6763zxg/AAA3-jPaRlYHMvsMyRbtRRmaa?dl=0

**北航网盘下载**：https://bhpan.buaa.edu.cn/link/AA84F755C78F1F4062BB81EBD5B41D5F7A

### 数据集结构

下载后请确保数据集按以下结构组织：

```
1-Digit-TrainSet/          # 手写数字训练集
├── 0_1.bmp
├── 0_2.bmp
└── ...

1-Digit-TestSet/           # 手写数字测试集
├── 0_1.bmp
├── 0_2.bmp
└── ...

2-MedImage-TrainSet/       # 医学图像训练集
├── disease_001.jpg
├── normal_001.jpg
└── ...

2-MedImage-TestSet/        # 医学图像测试集
├── disease_001.jpg
├── normal_001.jpg
└── ...

3-Saliency-TrainSet/       # 显著性训练集
├── Stimuli/
│   ├── Action/
│   ├── Affective/
│   └── ...
└── FIXATIONMAPS/
    ├── Action/
    ├── Affective/
    └── ...

3-Saliency-TestSet/        # 显著性测试集
├── Stimuli/
└── FIXATIONMAPS/

# CIFAR-10数据集会自动下载到 ./CIFARdata/
```

## 使用说明

### 题目一：手写数字识别

```bash
cd problem1
python digit_recognition.py --train_dir ../1-Digit-TrainSet --test_dir ../1-Digit-TestSet
```

**可选参数：**
- `--batch_size`: 批大小（默认64）
- `--num_epochs`: 训练轮数（默认20）
- `--lr`: 学习率（默认0.001）

### 题目二：医学图像检测

```bash
cd problem2
python medical_detection.py --train_dir ../2-MedImage-TrainSet --test_dir ../2-MedImage-TestSet
```

**可选参数：**
- `--batch_size`: 批大小（默认32）
- `--num_epochs`: 训练轮数（默认25）
- `--lr`: 学习率（默认0.0001）
- `--model_type`: 模型类型（simple/resnet，默认simple）

### 题目三：图像显著性预测

```bash
cd problem3
python saliency_prediction.py --train_dir ../3-Saliency-TrainSet --test_dir ../3-Saliency-TestSet
```

**可选参数：**
- `--batch_size`: 批大小（默认8）
- `--num_epochs`: 训练轮数（默认50）
- `--lr`: 学习率（默认0.001）

### 题目四：彩色图像生成

```bash
cd problem4
python image_generation.py
```

**可选参数：**
- `--batch_size`: 批大小（默认64）
- `--num_epochs`: 训练轮数（默认100）
- `--lr`: 学习率（默认0.0002）
- `--nz`: 噪声维度（默认100）

## 报告撰写

### LaTeX模板使用

每个题目都提供了中文和西班牙语版本的LaTeX报告模板：

1. **中文模板**：
   - `report_template_chinese.tex`（题目一）
   - `report_template_chinese_problem2.tex`（题目二）
   - `report_template_chinese_problem3.tex`（题目三）
   - `report_template_chinese_problem4.tex`（题目四）

2. **西班牙语模板**：
   - `report_template_spanish.tex`（题目一）
   - `report_template_spanish_problem2.tex`（题目二）
   - `report_template_spanish_problem3.tex`（题目三）
   - `report_template_spanish_problem4.tex`（题目四）

### 编译LaTeX文档

```bash
# 安装TeX Live（如果尚未安装）
# Ubuntu/Debian:
sudo apt-get install texlive-full

# macOS:
brew install --cask mactex

# Windows:
# 下载并安装 MiKTeX 或 TeX Live

# 编译文档
cd templates
xelatex report_template_chinese.tex
bibtex report_template_chinese
xelatex report_template_chinese.tex
xelatex report_template_chinese.tex
```

### 报告内容要求

根据课程要求，实验报告需要包含：
1. **问题描述** - 任务背景和目标
2. **实验模型原理和概述** - 理论基础和方法选择
3. **实验模型结构和参数** - 网络架构和超参数
4. **实验结果分析** - 训练和测试结果，包括失败案例分析
5. **总结** - 主要成果、局限性和改进方向

## 输出结果

每个程序运行后会生成以下输出：

### 题目一输出
- `problem1/training_curves.png` - 训练曲线
- `problem1/confusion_matrix.png` - 混淆矩阵
- `problem1/predictions.png` - 预测结果可视化
- `problem1/digit_model.pth` - 训练好的模型

### 题目二输出
- `problem2/training_curves.png` - 训练曲线
- `problem2/confusion_matrix.png` - 混淆矩阵
- `problem2/roc_curve.png` - ROC曲线
- `problem2/predictions.png` - 预测结果可视化
- `problem2/medical_model_*.pth` - 训练好的模型

### 题目三输出
- `problem3/training_curves.png` - 训练曲线
- `problem3/predictions.png` - 预测结果对比
- `problem3/saliency_model.pth` - 训练好的模型

### 题目四输出
- `problem4/training_progress.png` - 训练进度
- `problem4/real_vs_fake.png` - 真实vs生成图像对比
- `problem4/class_specific_samples.png` - 类别特定样本
- `problem4/generated_images_epoch_*.png` - 不同epoch的生成图像
- `problem4/gan_model.pth` - 训练好的GAN模型

## 性能基准

### 预期性能指标

| 题目 | 主要指标 | 预期性能 |
|------|----------|----------|
| 1. 手写数字识别 | 准确率 | > 98% |
| 2. 医学图像检测 | AUC | > 0.95 |
| 3. 图像显著性预测 | 相关系数(CC) | > 0.75 |
| 4. 彩色图像生成 | Inception Score | > 6.0 |

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减小批大小
   python script.py --batch_size 16
   ```

2. **数据路径错误**
   ```bash
   # 检查数据路径是否正确
   ls -la 1-Digit-TrainSet/
   ```

3. **依赖包版本冲突**
   ```bash
   # 重新创建虚拟环境
   conda create -n ml-assignment python=3.8
   conda activate ml-assignment
   pip install -r requirements.txt
   ```

4. **LaTeX编译错误**
   ```bash
   # 确保安装了中文支持
   sudo apt-get install texlive-lang-chinese
   ```

### 性能优化建议

1. **使用GPU加速**：确保安装了CUDA版本的PyTorch
2. **数据预处理**：将常用数据转换为更高效的格式
3. **批大小调整**：根据GPU内存调整批大小
4. **学习率调度**：使用学习率调度器提高收敛速度

## 贡献指南

欢迎提交Issue和Pull Request来改进本项目：

1. Fork本项目
2. 创建特性分支：`git checkout -b feature/new-feature`
3. 提交更改：`git commit -am 'Add new feature'`
4. 推送分支：`git push origin feature/new-feature`
5. 提交Pull Request

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至课程助教

## 致谢

感谢所有为本项目做出贡献的同学和老师。

---

**注意**：请确保在使用前阅读并理解所有代码，遵守学术诚信原则。