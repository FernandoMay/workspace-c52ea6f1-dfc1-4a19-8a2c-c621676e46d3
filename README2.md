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

python digit_recognition.py --train_dir ../TrainSet --test_dir ../TestSet

```

## Log Results

python digit_recognition.py --train_dir ../sets/TrainingSet --test_dir ../sets/TestSet
Using device: cpu
Training samples: 48000
Validation samples: 12000
Test samples: 10000
开始训练...
Epoch 1/20: 100%|█████████████████████████████████████████████████████████████████████| 750/750 [00:47<00:00, 15.89it/s]
Epoch 1: Train Loss: 0.1867, Train Acc: 94.04%, Val Loss: 0.0545, Val Acc: 98.21%
Epoch 2/20: 100%|█████████████████████████████████████████████████████████████████████| 750/750 [00:47<00:00, 15.76it/s]
Epoch 2: Train Loss: 0.0644, Train Acc: 97.95%, Val Loss: 0.0423, Val Acc: 98.75%
Epoch 3/20: 100%|█████████████████████████████████████████████████████████████████████| 750/750 [00:49<00:00, 15.22it/s]
Epoch 3: Train Loss: 0.0461, Train Acc: 98.58%, Val Loss: 0.0325, Val Acc: 99.11%
Epoch 4/20: 100%|█████████████████████████████████████████████████████████████████████| 750/750 [00:56<00:00, 13.25it/s]
Epoch 4: Train Loss: 0.0374, Train Acc: 98.80%, Val Loss: 0.0321, Val Acc: 99.12%
Epoch 5/20: 100%|█████████████████████████████████████████████████████████████████████| 750/750 [00:48<00:00, 15.31it/s]
Epoch 5: Train Loss: 0.0323, Train Acc: 99.01%, Val Loss: 0.0333, Val Acc: 98.99%
Epoch 6/20: 100%|█████████████████████████████████████████████████████████████████████| 750/750 [00:48<00:00, 15.42it/s]
Epoch 6: Train Loss: 0.0280, Train Acc: 99.14%, Val Loss: 0.0345, Val Acc: 99.00%
Epoch 7/20: 100%|█████████████████████████████████████████████████████████████████████| 750/750 [00:50<00:00, 14.88it/s]
Epoch 7: Train Loss: 0.0262, Train Acc: 99.16%, Val Loss: 0.0341, Val Acc: 99.08%
Epoch 8/20: 100%|█████████████████████████████████████████████████████████████████████| 750/750 [01:26<00:00,  8.69it/s]
Epoch 8: Train Loss: 0.0245, Train Acc: 99.24%, Val Loss: 0.0311, Val Acc: 99.20%
Epoch 9/20: 100%|█████████████████████████████████████████████████████████████████████| 750/750 [00:49<00:00, 15.21it/s]
Epoch 9: Train Loss: 0.0227, Train Acc: 99.34%, Val Loss: 0.0304, Val Acc: 99.11%
Epoch 10/20: 100%|████████████████████████████████████████████████████████████████████| 750/750 [00:48<00:00, 15.60it/s]
Epoch 10: Train Loss: 0.0163, Train Acc: 99.45%, Val Loss: 0.0339, Val Acc: 99.13%
Epoch 11/20: 100%|████████████████████████████████████████████████████████████████████| 750/750 [00:50<00:00, 14.96it/s]
Epoch 11: Train Loss: 0.0188, Train Acc: 99.40%, Val Loss: 0.0269, Val Acc: 99.28%
Epoch 12/20: 100%|████████████████████████████████████████████████████████████████████| 750/750 [01:20<00:00,  9.27it/s]
Epoch 12: Train Loss: 0.0165, Train Acc: 99.50%, Val Loss: 0.0366, Val Acc: 99.13%
Epoch 13/20: 100%|████████████████████████████████████████████████████████████████████| 750/750 [00:56<00:00, 13.24it/s]
Epoch 13: Train Loss: 0.0163, Train Acc: 99.49%, Val Loss: 0.0455, Val Acc: 99.01%
Epoch 14/20: 100%|████████████████████████████████████████████████████████████████████| 750/750 [00:55<00:00, 13.59it/s]
Epoch 14: Train Loss: 0.0157, Train Acc: 99.54%, Val Loss: 0.0431, Val Acc: 99.08%
Epoch 15/20: 100%|████████████████████████████████████████████████████████████████████| 750/750 [00:55<00:00, 13.45it/s]
Epoch 15: Train Loss: 0.0129, Train Acc: 99.60%, Val Loss: 0.0318, Val Acc: 99.12%
Epoch 16/20: 100%|████████████████████████████████████████████████████████████████████| 750/750 [00:59<00:00, 12.70it/s]
Epoch 16: Train Loss: 0.0142, Train Acc: 99.58%, Val Loss: 0.0343, Val Acc: 99.16%
Epoch 17/20: 100%|████████████████████████████████████████████████████████████████████| 750/750 [00:59<00:00, 12.54it/s]
Epoch 17: Train Loss: 0.0123, Train Acc: 99.65%, Val Loss: 0.0380, Val Acc: 99.07%
Epoch 18/20: 100%|████████████████████████████████████████████████████████████████████| 750/750 [01:02<00:00, 11.93it/s]
Epoch 18: Train Loss: 0.0153, Train Acc: 99.57%, Val Loss: 0.0379, Val Acc: 99.12%
Epoch 19/20: 100%|████████████████████████████████████████████████████████████████████| 750/750 [01:39<00:00,  7.56it/s]
Epoch 19: Train Loss: 0.0107, Train Acc: 99.68%, Val Loss: 0.0380, Val Acc: 99.21%
Epoch 20/20: 100%|████████████████████████████████████████████████████████████████████| 750/750 [01:01<00:00, 12.28it/s]
Epoch 20: Train Loss: 0.0119, Train Acc: 99.64%, Val Loss: 0.0439, Val Acc: 99.03%
在测试集上评估...
测试集准确率: 0.9921

分类报告:
              precision    recall  f1-score   support

           0     0.9969    0.9949    0.9959       980
           1     0.9939    0.9982    0.9960      1135
           2     0.9875    0.9961    0.9918      1032
           3     0.9960    0.9851    0.9905      1010
           4     0.9979    0.9857    0.9918       982
           5     0.9855    0.9910    0.9883       892
           6     0.9958    0.9906    0.9932       958
           7     0.9883    0.9893    0.9888      1028
           8     0.9939    0.9959    0.9949       974
           9     0.9853    0.9931    0.9891      1009

    accuracy                         0.9921     10000
   macro avg     0.9921    0.9920    0.9920     10000
weighted avg     0.9921    0.9921    0.9921     10000

模型已保存到 problem1/digit_model.pth

**可选参数：**
- `--batch_size`: 批大小（默认64）
- `--num_epochs`: 训练轮数（默认20）
- `--lr`: 学习率（默认0.001）

### 题目二：医学图像检测

```bash
cd problem2
python medical_detection.py --train_dir ../sets/2-MedImage-TrainSet --test_dir ../sets/2-MedImage-TestSet
```

## Log Results

ython medical_detection.py --train_dir ../sets/2-MedImage-TrainSet --test_dir ../sets/2-MedImage-TestSet
Using device: cpu
Training samples: 1311
Validation samples: 328
Test samples: 250
训练集类别分布: Counter({0: 777, 1: 534})
medical_detection.py:387: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
开始训练...
Epoch 1/25: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 41/41 [01:31<00:00,  2.23s/it]
Epoch 1: Train Loss: 0.7796, Train Acc: 55.30%, Val Loss: 0.6141, Val Acc: 77.44%, Val AUC: 0.8191
Epoch 2/25: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 41/41 [01:31<00:00,  2.23s/it]
Epoch 2: Train Loss: 0.6574, Train Acc: 71.93%, Val Loss: 0.5649, Val Acc: 78.66%, Val AUC: 0.8438
Epoch 3/25: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 41/41 [01:29<00:00,  2.19s/it]
Epoch 3: Train Loss: 0.6042, Train Acc: 76.13%, Val Loss: 0.5622, Val Acc: 75.30%, Val AUC: 0.8576
Epoch 4/25: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 41/41 [01:36<00:00,  2.36s/it]
Epoch 4: Train Loss: 0.5711, Train Acc: 78.18%, Val Loss: 0.5383, Val Acc: 78.35%, Val AUC: 0.8815
Epoch 5/25: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 41/41 [01:39<00:00,  2.42s/it]
Epoch 5: Train Loss: 0.5458, Train Acc: 78.49%, Val Loss: 0.5137, Val Acc: 80.18%, Val AUC: 0.8791
Epoch 6/25: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 41/41 [01:33<00:00,  2.28s/it]
Epoch 6: Train Loss: 0.5030, Train Acc: 79.86%, Val Loss: 0.4498, Val Acc: 83.23%, Val AUC: 0.9022
Epoch 7/25: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 41/41 [01:45<00:00,  2.58s/it]
Epoch 7: Train Loss: 0.4788, Train Acc: 82.46%, Val Loss: 0.4569, Val Acc: 80.79%, Val AUC: 0.9040
Epoch 8/25: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 41/41 [01:41<00:00,  2.47s/it]
Epoch 8: Train Loss: 0.4592, Train Acc: 82.30%, Val Loss: 0.5269, Val Acc: 82.93%, Val AUC: 0.8960
Epoch 9/25: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 41/41 [01:44<00:00,  2.56s/it]
Epoch 9: Train Loss: 0.4684, Train Acc: 82.46%, Val Loss: 0.5230, Val Acc: 81.40%, Val AUC: 0.8894
Epoch 10/25: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 41/41 [01:36<00:00,  2.36s/it]
Epoch 10: Train Loss: 0.4353, Train Acc: 84.13%, Val Loss: 0.4418, Val Acc: 84.45%, Val AUC: 0.9202
Epoch 11/25: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 41/41 [01:30<00:00,  2.21s/it]
Epoch 11: Train Loss: 0.3964, Train Acc: 86.42%, Val Loss: 0.3968, Val Acc: 86.28%, Val AUC: 0.9342
Epoch 12/25: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 41/41 [01:29<00:00,  2.18s/it]
Epoch 12: Train Loss: 0.3820, Train Acc: 87.19%, Val Loss: 0.4044, Val Acc: 85.37%, Val AUC: 0.9315
Epoch 13/25: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 41/41 [01:31<00:00,  2.23s/it]
Epoch 13: Train Loss: 0.3752, Train Acc: 87.41%, Val Loss: 0.4371, Val Acc: 86.28%, Val AUC: 0.9198
Epoch 14/25: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 41/41 [01:35<00:00,  2.32s/it]
Epoch 14: Train Loss: 0.3741, Train Acc: 86.50%, Val Loss: 0.4328, Val Acc: 87.20%, Val AUC: 0.9298
Epoch 15/25: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 41/41 [01:30<00:00,  2.21s/it]
Epoch 15: Train Loss: 0.3350, Train Acc: 88.63%, Val Loss: 0.4405, Val Acc: 84.45%, Val AUC: 0.9341
Epoch 16/25: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 41/41 [01:34<00:00,  2.30s/it]
Epoch 16: Train Loss: 0.3008, Train Acc: 90.01%, Val Loss: 0.5007, Val Acc: 87.80%, Val AUC: 0.9280
Epoch 17/25: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 41/41 [01:35<00:00,  2.32s/it]
Epoch 17: Train Loss: 0.3160, Train Acc: 89.55%, Val Loss: 0.5475, Val Acc: 88.11%, Val AUC: 0.9188
Epoch 18/25: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 41/41 [01:30<00:00,  2.20s/it]
Epoch 18: Train Loss: 0.3152, Train Acc: 89.55%, Val Loss: 0.4030, Val Acc: 87.20%, Val AUC: 0.9427
Epoch 19/25: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 41/41 [01:28<00:00,  2.16s/it]
Epoch 19: Train Loss: 0.3048, Train Acc: 90.08%, Val Loss: 0.3938, Val Acc: 89.33%, Val AUC: 0.9416
Epoch 20/25: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 41/41 [01:30<00:00,  2.20s/it]
Epoch 20: Train Loss: 0.2836, Train Acc: 90.24%, Val Loss: 0.3633, Val Acc: 89.33%, Val AUC: 0.9525
Epoch 21/25: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 41/41 [01:30<00:00,  2.21s/it]
Epoch 21: Train Loss: 0.2733, Train Acc: 90.92%, Val Loss: 0.3887, Val Acc: 91.46%, Val AUC: 0.9456
Epoch 22/25: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 41/41 [01:30<00:00,  2.20s/it]
Epoch 22: Train Loss: 0.2579, Train Acc: 91.84%, Val Loss: 0.4375, Val Acc: 88.72%, Val AUC: 0.9488
Epoch 23/25: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 41/41 [01:27<00:00,  2.13s/it]
Epoch 23: Train Loss: 0.2548, Train Acc: 92.45%, Val Loss: 0.3613, Val Acc: 89.63%, Val AUC: 0.9575
Epoch 24/25: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 41/41 [01:30<00:00,  2.21s/it]
Epoch 24: Train Loss: 0.2367, Train Acc: 92.07%, Val Loss: 0.3061, Val Acc: 90.55%, Val AUC: 0.9661
Epoch 25/25: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 41/41 [01:26<00:00,  2.12s/it]
Epoch 25: Train Loss: 0.2292, Train Acc: 92.45%, Val Loss: 0.3870, Val Acc: 89.02%, Val AUC: 0.9499
在测试集上评估...
测试集准确率: 0.7360
测试集AUC: 0.8748

分类报告:
              precision    recall  f1-score   support

      Normal     0.9773    0.5733    0.7227       150
     Disease     0.6049    0.9800    0.7481       100

    accuracy                         0.7360       250
   macro avg     0.7911    0.7767    0.7354       250
weighted avg     0.8283    0.7360    0.7329       250


分析失败案例...
失败案例数量: 66

失败案例示例:
案例 1: 真实=Disease, 预测=Normal, 概率=0.388
案例 2: 真实=Disease, 预测=Normal, 概率=0.268
案例 3: 真实=Normal, 预测=Disease, 概率=0.741
案例 4: 真实=Normal, 预测=Disease, 概率=0.909
案例 5: 真实=Normal, 预测=Disease, 概率=0.999
模型已保存到 problem2/medical_model_simple.pth

**可选参数：**
- `--batch_size`: 批大小（默认32）
- `--num_epochs`: 训练轮数（默认25）
- `--lr`: 学习率（默认0.0001）
- `--model_type`: 模型类型（simple/resnet，默认simple）

### 题目三：图像显著性预测

```bash
cd problem3
python saliency_prediction.py --train_dir ../sets/3-Saliency-TrainSet --test_dir ../sets/3-Saliency-TestSet
```

## Log Results I

python saliency_prediction.py --train_dir ../sets/3-Saliency-TrainSet --test_dir ../sets/3-Saliency-TestSet
Using device: cpu
Training samples: 1280
Validation samples: 320
Test samples: 400
开始训练...
Epoch 1/50: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 160/160 [26:33<00:00,  9.96s/it]
Epoch 1: Train Loss: -0.1757, Train MSE: 0.0449, Train CC: 0.3964, Val Loss: -0.2264, Val MSE: 0.0119, Val CC: 0.4648
Epoch 2/50: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 160/160 [25:57<00:00,  9.73s/it]
Epoch 2: Train Loss: -0.2342, Train MSE: 0.0114, Train CC: 0.4798, Val Loss: -0.2390, Val MSE: 0.0103, Val CC: 0.4884
Epoch 3/50: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 160/160 [22:16<00:00,  8.35s/it]
Epoch 3: Train Loss: -0.2392, Train MSE: 0.0106, Train CC: 0.4890, Val Loss: -0.2424, Val MSE: 0.0102, Val CC: 0.4949
Epoch 4/50: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 160/160 [22:41<00:00,  8.51s/it]
Epoch 4: Train Loss: -0.2570, Train MSE: 0.0100, Train CC: 0.5240, Val Loss: -0.2530, Val MSE: 0.0103, Val CC: 0.5164
Epoch 5/50: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 160/160 [22:02<00:00,  8.27s/it]
Epoch 5: Train Loss: -0.2638, Train MSE: 0.0098, Train CC: 0.5373, Val Loss: -0.2662, Val MSE: 0.0095, Val CC: 0.5419
Epoch 6/50: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 160/160 [23:00<00:00,  8.63s/it]
Epoch 6: Train Loss: -0.2669, Train MSE: 0.0096, Train CC: 0.5434, Val Loss: -0.2672, Val MSE: 0.0094, Val CC: 0.5438
Epoch 7/50: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 160/160 [20:43<00:00,  7.77s/it]
Epoch 7: Train Loss: -0.2674, Train MSE: 0.0096, Train CC: 0.5444, Val Loss: -0.2624, Val MSE: 0.0162, Val CC: 0.5410
Epoch 8/50: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 160/160 [20:33<00:00,  7.71s/it]
Epoch 8: Train Loss: -0.2702, Train MSE: 0.0095, Train CC: 0.5500, Val Loss: -0.2646, Val MSE: 0.0097, Val CC: 0.5389
Epoch 9/50: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 160/160 [23:45<00:00,  8.91s/it]
Epoch 9: Train Loss: -0.2703, Train MSE: 0.0096, Train CC: 0.5501, Val Loss: -0.2641, Val MSE: 0.0185, Val CC: 0.5466
Epoch 10/50: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 160/160 [21:35<00:00,  8.10s/it]
Epoch 10: Train Loss: -0.2737, Train MSE: 0.0094, Train CC: 0.5569, Val Loss: -0.2714, Val MSE: 0.0093, Val CC: 0.5521
Epoch 11/50: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 160/160 [23:06<00:00,  8.67s/it]
Epoch 11: Train Loss: -0.2733, Train MSE: 0.0094, Train CC: 0.5560, Val Loss: -0.2725, Val MSE: 0.0093, Val CC: 0.5542
Epoch 12/50: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 160/160 [23:15<00:00,  8.72s/it]
Epoch 12: Train Loss: -0.2742, Train MSE: 0.0094, Train CC: 0.5578, Val Loss: -0.2685, Val MSE: 0.0093, Val CC: 0.5463
Epoch 13/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 160/160 [1:33:36<00:00, 35.10s/it]
Epoch 13: Train Loss: -0.2767, Train MSE: 0.0093, Train CC: 0.5627, Val Loss: -0.2667, Val MSE: 0.0097, Val CC: 0.5431
Epoch 14/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 160/160 [1:22:06<00:00, 30.79s/it]
Epoch 14: Train Loss: -0.2788, Train MSE: 0.0093, Train CC: 0.5670, Val Loss: -0.2690, Val MSE: 0.0099, Val CC: 0.5479
Epoch 15/50: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 160/160 [36:47<00:00, 13.80s/it]
Epoch 15: Train Loss: -0.2803, Train MSE: 0.0092, Train CC: 0.5697, Val Loss: -0.2755, Val MSE: 0.0091, Val CC: 0.5601
Epoch 16/50: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 160/160 [23:00<00:00,  8.63s/it]
Epoch 16: Train Loss: -0.2810, Train MSE: 0.0092, Train CC: 0.5712, Val Loss: -0.2736, Val MSE: 0.0094, Val CC: 0.5566
Epoch 17/50: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 160/160 [40:43<00:00, 15.27s/it]
Epoch 17: Train Loss: -0.2828, Train MSE: 0.0092, Train CC: 0.5748, Val Loss: -0.2693, Val MSE: 0.0103, Val CC: 0.5489
Epoch 18/50: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 160/160 [48:36<00:00, 18.23s/it]
Epoch 18: Train Loss: -0.2863, Train MSE: 0.0091, Train CC: 0.5817, Val Loss: -0.2697, Val MSE: 0.0093, Val CC: 0.5486
Epoch 19/50: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 160/160 [24:16<00:00,  9.10s/it]
Epoch 19: Train Loss: -0.2883, Train MSE: 0.0089, Train CC: 0.5856, Val Loss: -0.2696, Val MSE: 0.0093, Val CC: 0.5485
Epoch 20/50: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 160/160 [40:47<00:00, 15.30s/it]
Epoch 20: Train Loss: -0.2906, Train MSE: 0.0089, Train CC: 0.5901, Val Loss: -0.2644, Val MSE: 0.0095, Val CC: 0.5382
Epoch 21/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 160/160 [1:08:12<00:00, 25.58s/it]
Epoch 21: Train Loss: -0.2945, Train MSE: 0.0087, Train CC: 0.5978, Val Loss: -0.2679, Val MSE: 0.0094, Val CC: 0.5451
Epoch 22/50: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 160/160 [23:21<00:00,  8.76s/it]
Epoch 22: Train Loss: -0.2991, Train MSE: 0.0086, Train CC: 0.6068, Val Loss: -0.2679, Val MSE: 0.0093, Val CC: 0.5451
Epoch 23/50: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 160/160 [25:15<00:00,  9.47s/it]
Epoch 23: Train Loss: -0.3022, Train MSE: 0.0085, Train CC: 0.6129, Val Loss: -0.2714, Val MSE: 0.0092, Val CC: 0.5520
Epoch 24/50: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 160/160 [51:46<00:00, 19.42s/it]
Epoch 24: Train Loss: -0.3101, Train MSE: 0.0083, Train CC: 0.6285, Val Loss: -0.2673, Val MSE: 0.0094, Val CC: 0.5439
Epoch 25/50: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 160/160 [23:12<00:00,  8.70s/it]
Epoch 25: Train Loss: -0.3191, Train MSE: 0.0079, Train CC: 0.6461, Val Loss: -0.2664, Val MSE: 0.0094, Val CC: 0.5422
Epoch 26/50: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 160/160 [1:00:20<00:00, 22.63s/it]
Epoch 26: Train Loss: -0.3283, Train MSE: 0.0076, Train CC: 0.6642, Val Loss: -0.2631, Val MSE: 0.0094, Val CC: 0.5356
Epoch 27/50: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 160/160 [21:14<00:00,  7.97s/it]
Epoch 27: Train Loss: -0.3402, Train MSE: 0.0073, Train CC: 0.6876, Val Loss: -0.2275, Val MSE: 0.0112, Val CC: 0.4663

## Log Results II

python saliency_prediction.py --train_dir ../sets/3-Saliency-TrainSet --test_dir ../sets/3-Saliency-TestSet
Using device: cpu
Training samples: 1280
Validation samples: 320
Test samples: 400
开始训练...
Epoch 1/50: 100%|███████████████████████████████████████████████████████████████████████████████████████| 160/160 [26:24<00:00,  9.91s/it]
Epoch 1: Train Loss: -0.1572, Train MSE: 0.0503, Train CC: 0.3646, Val Loss: -0.2121, Val MSE: 0.0131, Val CC: 0.4373
Epoch 2/50: 100%|███████████████████████████████████████████████████████████████████████████████████████| 160/160 [23:48<00:00,  8.93s/it]
Epoch 2: Train Loss: -0.2310, Train MSE: 0.0124, Train CC: 0.4744, Val Loss: -0.2343, Val MSE: 0.0131, Val CC: 0.4816
Epoch 3/50: 100%|███████████████████████████████████████████████████████████████████████████████████████| 160/160 [25:04<00:00,  9.41s/it]
Epoch 3: Train Loss: -0.2519, Train MSE: 0.0103, Train CC: 0.5141, Val Loss: -0.2621, Val MSE: 0.0109, Val CC: 0.5350
Epoch 4/50: 100%|███████████████████████████████████████████████████████████████████████████████████████| 160/160 [28:29<00:00, 10.68s/it]
Epoch 4: Train Loss: -0.2585, Train MSE: 0.0099, Train CC: 0.5269, Val Loss: -0.2590, Val MSE: 0.0110, Val CC: 0.5290
Epoch 5/50: 100%|███████████████████████████████████████████████████████████████████████████████████████| 160/160 [29:15<00:00, 10.97s/it]
Epoch 5: Train Loss: -0.2636, Train MSE: 0.0097, Train CC: 0.5368, Val Loss: -0.2641, Val MSE: 0.0116, Val CC: 0.5398
Epoch 6/50: 100%|███████████████████████████████████████████████████████████████████████████████████████| 160/160 [26:50<00:00, 10.06s/it]
Epoch 6: Train Loss: -0.2654, Train MSE: 0.0096, Train CC: 0.5403, Val Loss: -0.2645, Val MSE: 0.0098, Val CC: 0.5387
Epoch 7/50: 100%|███████████████████████████████████████████████████████████████████████████████████████| 160/160 [26:19<00:00,  9.87s/it]
Epoch 7: Train Loss: -0.2657, Train MSE: 0.0095, Train CC: 0.5409, Val Loss: -0.2650, Val MSE: 0.0098, Val CC: 0.5397
Epoch 8/50: 100%|███████████████████████████████████████████████████████████████████████████████████████| 160/160 [25:32<00:00,  9.58s/it]
Epoch 8: Train Loss: -0.2689, Train MSE: 0.0094, Train CC: 0.5472, Val Loss: -0.2602, Val MSE: 0.0100, Val CC: 0.5303
Epoch 9/50: 100%|███████████████████████████████████████████████████████████████████████████████████████| 160/160 [25:43<00:00,  9.65s/it]
Epoch 9: Train Loss: -0.2686, Train MSE: 0.0095, Train CC: 0.5466, Val Loss: -0.2578, Val MSE: 0.0106, Val CC: 0.5262
Epoch 10/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [25:27<00:00,  9.54s/it]
Epoch 10: Train Loss: -0.2701, Train MSE: 0.0094, Train CC: 0.5497, Val Loss: -0.2623, Val MSE: 0.0098, Val CC: 0.5345
Epoch 11/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [25:55<00:00,  9.72s/it]
Epoch 11: Train Loss: -0.2726, Train MSE: 0.0093, Train CC: 0.5545, Val Loss: -0.2709, Val MSE: 0.0096, Val CC: 0.5513
Epoch 12/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [26:18<00:00,  9.87s/it]
Epoch 12: Train Loss: -0.2748, Train MSE: 0.0093, Train CC: 0.5590, Val Loss: -0.2642, Val MSE: 0.0097, Val CC: 0.5382
Epoch 13/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [26:33<00:00,  9.96s/it]
Epoch 13: Train Loss: -0.2743, Train MSE: 0.0093, Train CC: 0.5579, Val Loss: -0.2685, Val MSE: 0.0097, Val CC: 0.5466
Epoch 14/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [26:37<00:00,  9.99s/it]
Epoch 14: Train Loss: -0.2755, Train MSE: 0.0092, Train CC: 0.5602, Val Loss: -0.2716, Val MSE: 0.0096, Val CC: 0.5528
Epoch 15/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [28:39<00:00, 10.75s/it]
Epoch 15: Train Loss: -0.2759, Train MSE: 0.0092, Train CC: 0.5611, Val Loss: -0.2671, Val MSE: 0.0097, Val CC: 0.5439
Epoch 16/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [28:46<00:00, 10.79s/it]
Epoch 16: Train Loss: -0.2801, Train MSE: 0.0091, Train CC: 0.5693, Val Loss: -0.2714, Val MSE: 0.0096, Val CC: 0.5524
Epoch 17/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [29:58<00:00, 11.24s/it]
Epoch 17: Train Loss: -0.2804, Train MSE: 0.0091, Train CC: 0.5698, Val Loss: -0.2693, Val MSE: 0.0097, Val CC: 0.5482
Epoch 18/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [30:11<00:00, 11.32s/it]
Epoch 18: Train Loss: -0.2820, Train MSE: 0.0090, Train CC: 0.5731, Val Loss: -0.2712, Val MSE: 0.0095, Val CC: 0.5519
Epoch 19/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [28:47<00:00, 10.80s/it]
Epoch 19: Train Loss: -0.2823, Train MSE: 0.0090, Train CC: 0.5737, Val Loss: -0.2731, Val MSE: 0.0097, Val CC: 0.5558
Epoch 20/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [27:53<00:00, 10.46s/it]
Epoch 20: Train Loss: -0.2856, Train MSE: 0.0089, Train CC: 0.5802, Val Loss: -0.2720, Val MSE: 0.0095, Val CC: 0.5535
Epoch 21/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [28:23<00:00, 10.65s/it]
Epoch 21: Train Loss: -0.2856, Train MSE: 0.0089, Train CC: 0.5802, Val Loss: -0.2671, Val MSE: 0.0100, Val CC: 0.5442
Epoch 22/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [28:11<00:00, 10.57s/it]
Epoch 22: Train Loss: -0.2869, Train MSE: 0.0089, Train CC: 0.5828, Val Loss: -0.2711, Val MSE: 0.0097, Val CC: 0.5519
Epoch 23/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [28:08<00:00, 10.55s/it]
Epoch 23: Train Loss: -0.2903, Train MSE: 0.0088, Train CC: 0.5893, Val Loss: -0.2664, Val MSE: 0.0099, Val CC: 0.5427
Epoch 24/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [28:33<00:00, 10.71s/it]
Epoch 24: Train Loss: -0.2932, Train MSE: 0.0087, Train CC: 0.5951, Val Loss: -0.2677, Val MSE: 0.0098, Val CC: 0.5451
Epoch 25/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [29:01<00:00, 10.88s/it]
Epoch 25: Train Loss: -0.2938, Train MSE: 0.0087, Train CC: 0.5963, Val Loss: -0.2661, Val MSE: 0.0097, Val CC: 0.5420
Epoch 26/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [29:55<00:00, 11.22s/it]
Epoch 26: Train Loss: -0.2982, Train MSE: 0.0085, Train CC: 0.6049, Val Loss: -0.2661, Val MSE: 0.0097, Val CC: 0.5419
Epoch 27/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [28:08<00:00, 10.55s/it]
Epoch 27: Train Loss: -0.3020, Train MSE: 0.0084, Train CC: 0.6125, Val Loss: -0.2697, Val MSE: 0.0096, Val CC: 0.5491
Epoch 28/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [23:58<00:00,  8.99s/it]
Epoch 28: Train Loss: -0.3053, Train MSE: 0.0083, Train CC: 0.6188, Val Loss: -0.2661, Val MSE: 0.0097, Val CC: 0.5419
Epoch 29/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [24:33<00:00,  9.21s/it]
Epoch 29: Train Loss: -0.3112, Train MSE: 0.0081, Train CC: 0.6306, Val Loss: -0.2575, Val MSE: 0.0108, Val CC: 0.5257
Epoch 30/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [27:22<00:00, 10.27s/it]
Epoch 30: Train Loss: -0.3144, Train MSE: 0.0080, Train CC: 0.6369, Val Loss: -0.2607, Val MSE: 0.0099, Val CC: 0.5312
Epoch 31/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [27:10<00:00, 10.19s/it]
Epoch 31: Train Loss: -0.3227, Train MSE: 0.0078, Train CC: 0.6533, Val Loss: -0.2636, Val MSE: 0.0098, Val CC: 0.5370
Epoch 32/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [28:11<00:00, 10.57s/it]
Epoch 32: Train Loss: -0.3301, Train MSE: 0.0075, Train CC: 0.6676, Val Loss: -0.2509, Val MSE: 0.0108, Val CC: 0.5127
Epoch 33/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [28:09<00:00, 10.56s/it]
Epoch 33: Train Loss: -0.3366, Train MSE: 0.0073, Train CC: 0.6804, Val Loss: -0.2565, Val MSE: 0.0128, Val CC: 0.5259
Epoch 34/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [25:02<00:00,  9.39s/it]
Epoch 34: Train Loss: -0.3460, Train MSE: 0.0070, Train CC: 0.6989, Val Loss: -0.2492, Val MSE: 0.0103, Val CC: 0.5088
Epoch 35/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [26:35<00:00,  9.97s/it]
Epoch 35: Train Loss: -0.3577, Train MSE: 0.0065, Train CC: 0.7219, Val Loss: -0.2493, Val MSE: 0.0113, Val CC: 0.5099
Epoch 36/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [24:25<00:00,  9.16s/it]
Epoch 36: Train Loss: -0.3667, Train MSE: 0.0062, Train CC: 0.7396, Val Loss: -0.2376, Val MSE: 0.0138, Val CC: 0.4890
Epoch 37/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [23:20<00:00,  8.75s/it]
Epoch 37: Train Loss: -0.3752, Train MSE: 0.0059, Train CC: 0.7563, Val Loss: -0.2386, Val MSE: 0.0110, Val CC: 0.4882
Epoch 38/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [24:06<00:00,  9.04s/it]
Epoch 38: Train Loss: -0.3877, Train MSE: 0.0054, Train CC: 0.7807, Val Loss: -0.2411, Val MSE: 0.0108, Val CC: 0.4931
Epoch 39/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [24:00<00:00,  9.01s/it]
Epoch 39: Train Loss: -0.3977, Train MSE: 0.0049, Train CC: 0.8004, Val Loss: -0.2323, Val MSE: 0.0108, Val CC: 0.4754
Epoch 40/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [24:00<00:00,  9.00s/it]
Epoch 40: Train Loss: -0.4078, Train MSE: 0.0045, Train CC: 0.8201, Val Loss: -0.2175, Val MSE: 0.0114, Val CC: 0.4465
Epoch 41/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [23:22<00:00,  8.77s/it]
Epoch 41: Train Loss: -0.4121, Train MSE: 0.0043, Train CC: 0.8285, Val Loss: -0.2289, Val MSE: 0.0140, Val CC: 0.4717
Epoch 42/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [22:58<00:00,  8.62s/it]
Epoch 42: Train Loss: -0.4200, Train MSE: 0.0040, Train CC: 0.8439, Val Loss: -0.2306, Val MSE: 0.0109, Val CC: 0.4721
Epoch 43/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [22:41<00:00,  8.51s/it]
Epoch 43: Train Loss: -0.4267, Train MSE: 0.0037, Train CC: 0.8571, Val Loss: -0.2225, Val MSE: 0.0110, Val CC: 0.4561
Epoch 44/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [23:17<00:00,  8.74s/it]
Epoch 44: Train Loss: -0.4335, Train MSE: 0.0033, Train CC: 0.8702, Val Loss: -0.2272, Val MSE: 0.0112, Val CC: 0.4656
Epoch 45/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [23:24<00:00,  8.78s/it]
Epoch 45: Train Loss: -0.4379, Train MSE: 0.0031, Train CC: 0.8789, Val Loss: -0.2323, Val MSE: 0.0116, Val CC: 0.4761
Epoch 46/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [26:36<00:00,  9.98s/it]
Epoch 46: Train Loss: -0.4444, Train MSE: 0.0028, Train CC: 0.8915, Val Loss: -0.2272, Val MSE: 0.0112, Val CC: 0.4656
Epoch 47/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [26:45<00:00, 10.04s/it]
Epoch 47: Train Loss: -0.4472, Train MSE: 0.0027, Train CC: 0.8970, Val Loss: -0.2308, Val MSE: 0.0109, Val CC: 0.4725
Epoch 48/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [23:29<00:00,  8.81s/it]
Epoch 48: Train Loss: -0.4504, Train MSE: 0.0025, Train CC: 0.9034, Val Loss: -0.2153, Val MSE: 0.0116, Val CC: 0.4422
Epoch 49/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [23:40<00:00,  8.88s/it]
Epoch 49: Train Loss: -0.4510, Train MSE: 0.0025, Train CC: 0.9045, Val Loss: -0.2276, Val MSE: 0.0111, Val CC: 0.4664
Epoch 50/50: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160/160 [23:46<00:00,  8.91s/it]
Epoch 50: Train Loss: -0.4525, Train MSE: 0.0024, Train CC: 0.9074, Val Loss: -0.2212, Val MSE: 0.0111, Val CC: 0.4535
在测试集上评估...
Evaluating:   0%|                                                                                                  | 0/50 [00:00<?, ?it/s]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:   2%|█▊                                                                                        | 1/50 [00:03<02:52,  3.52s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:   4%|███▌                                                                                      | 2/50 [00:06<02:25,  3.04s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:   6%|█████▍                                                                                    | 3/50 [00:08<02:16,  2.90s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:   8%|███████▏                                                                                  | 4/50 [00:11<02:11,  2.87s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  10%|█████████                                                                                 | 5/50 [00:14<02:09,  2.88s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  12%|██████████▊                                                                               | 6/50 [00:17<02:06,  2.88s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  14%|████████████▌                                                                             | 7/50 [00:20<02:00,  2.81s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  16%|██████████████▍                                                                           | 8/50 [00:22<01:56,  2.77s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  18%|████████████████▏                                                                         | 9/50 [00:25<01:53,  2.77s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  20%|█████████████████▊                                                                       | 10/50 [00:28<01:50,  2.77s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  22%|███████████████████▌                                                                     | 11/50 [00:31<01:49,  2.80s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  24%|█████████████████████▎                                                                   | 12/50 [00:34<01:46,  2.81s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  26%|███████████████████████▏                                                                 | 13/50 [00:36<01:43,  2.81s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  28%|████████████████████████▉                                                                | 14/50 [00:39<01:41,  2.83s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  30%|██████████████████████████▋                                                              | 15/50 [00:42<01:39,  2.85s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  32%|████████████████████████████▍                                                            | 16/50 [00:45<01:36,  2.83s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  34%|██████████████████████████████▎                                                          | 17/50 [00:48<01:33,  2.84s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  36%|████████████████████████████████                                                         | 18/50 [00:51<01:30,  2.83s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  38%|█████████████████████████████████▊                                                       | 19/50 [00:53<01:26,  2.80s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  40%|███████████████████████████████████▌                                                     | 20/50 [00:56<01:24,  2.81s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  42%|█████████████████████████████████████▍                                                   | 21/50 [00:59<01:20,  2.79s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  44%|███████████████████████████████████████▏                                                 | 22/50 [01:02<01:18,  2.81s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  46%|████████████████████████████████████████▉                                                | 23/50 [01:05<01:14,  2.78s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  48%|██████████████████████████████████████████▋                                              | 24/50 [01:07<01:12,  2.78s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  50%|████████████████████████████████████████████▌                                            | 25/50 [01:10<01:09,  2.80s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  52%|██████████████████████████████████████████████▎                                          | 26/50 [01:13<01:07,  2.83s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  54%|████████████████████████████████████████████████                                         | 27/50 [01:16<01:04,  2.82s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  56%|█████████████████████████████████████████████████▊                                       | 28/50 [01:19<01:02,  2.83s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  58%|███████████████████████████████████████████████████▌                                     | 29/50 [01:21<00:59,  2.82s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  60%|█████████████████████████████████████████████████████▍                                   | 30/50 [01:24<00:56,  2.80s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  62%|███████████████████████████████████████████████████████▏                                 | 31/50 [01:27<00:52,  2.78s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  64%|████████████████████████████████████████████████████████▉                                | 32/50 [01:30<00:50,  2.78s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  66%|██████████████████████████████████████████████████████████▋                              | 33/50 [01:33<00:47,  2.78s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  68%|████████████████████████████████████████████████████████████▌                            | 34/50 [01:35<00:44,  2.81s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  70%|██████████████████████████████████████████████████████████████▎                          | 35/50 [01:38<00:42,  2.82s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  72%|████████████████████████████████████████████████████████████████                         | 36/50 [01:41<00:39,  2.81s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  74%|█████████████████████████████████████████████████████████████████▊                       | 37/50 [01:44<00:36,  2.82s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  76%|███████████████████████████████████████████████████████████████████▋                     | 38/50 [01:47<00:33,  2.80s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  78%|█████████████████████████████████████████████████████████████████████▍                   | 39/50 [01:49<00:30,  2.78s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  80%|███████████████████████████████████████████████████████████████████████▏                 | 40/50 [01:52<00:27,  2.75s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  82%|████████████████████████████████████████████████████████████████████████▉                | 41/50 [01:55<00:24,  2.76s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  84%|██████████████████████████████████████████████████████████████████████████▊              | 42/50 [01:58<00:22,  2.78s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  86%|████████████████████████████████████████████████████████████████████████████▌            | 43/50 [02:00<00:19,  2.78s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  88%|██████████████████████████████████████████████████████████████████████████████▎          | 44/50 [02:03<00:16,  2.79s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  90%|████████████████████████████████████████████████████████████████████████████████         | 45/50 [02:06<00:13,  2.80s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  92%|█████████████████████████████████████████████████████████████████████████████████▉       | 46/50 [02:09<00:11,  2.80s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  94%|███████████████████████████████████████████████████████████████████████████████████▋     | 47/50 [02:12<00:08,  2.80s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  96%|█████████████████████████████████████████████████████████████████████████████████████▍   | 48/50 [02:14<00:05,  2.78s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating:  98%|███████████████████████████████████████████████████████████████████████████████████████▏ | 49/50 [02:17<00:02,  2.74s/it]saliency_prediction.py:226: RuntimeWarning: divide by zero encountered in log
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
saliency_prediction.py:226: RuntimeWarning: invalid value encountered in multiply
  kl_div = np.sum(target_prob * np.log(target_prob / pred_prob))
Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 50/50 [02:20<00:00,  2.81s/it]
测试集平均指标:
  mse: 0.0119
  cc: 0.4487
  kl_div: nan
  js_div: 0.1842
Traceback (most recent call last):
  File "saliency_prediction.py", line 508, in <module>
    main()
  File "saliency_prediction.py", line 498, in main
    visualize_predictions(predictions, targets)
  File "saliency_prediction.py", line 393, in visualize_predictions
    axes[i, 0].imshow(target, cmap='gray')
  File "/opt/homebrew/Caskroom/miniconda/base/envs/ml-assignment/lib/python3.8/site-packages/matplotlib/__init__.py", line 1446, in inner
    return func(ax, *map(sanitize_sequence, args), **kwargs)
  File "/opt/homebrew/Caskroom/miniconda/base/envs/ml-assignment/lib/python3.8/site-packages/matplotlib/axes/_axes.py", line 5663, in imshow
    im.set_data(X)
  File "/opt/homebrew/Caskroom/miniconda/base/envs/ml-assignment/lib/python3.8/site-packages/matplotlib/image.py", line 710, in set_data
    raise TypeError("Invalid shape {} for image data"
TypeError: Invalid shape (1, 256, 256) for image data


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