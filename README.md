# 脑部MRI分类深度学习模型

## 项目结构
```text
brain_mri_classification/
├── data/
│   ├── raw/                # 原始数据
│   ├── processed/          # 预处理后的数据
│   └── splits/             # train/val/test 划分文件
│
├── datasets/
│   ├── __init__.py
│   └── brain_mri_dataset.py
│
├── models/
│   ├── __init__.py
│   └── simple_cnn.py
│
├── utils/
│   ├── __init__.py
│   ├── io.py               # 读 nii / 图像
│   ├── preprocess.py       # 预处理
│   └── metrics.py
│
├── configs/
│   └── base.yaml
│
├── train.py
├── eval.py
├── infer.py
│
├── requirements.txt
└── README.md
```

## 数据预处理
原始数据的路径是/home/tbing/projects/data/brainMRI，该目录下有“脑膜病变图像”“正常头颅MRI”和“MASK”三个子目录。由于首先处理的是脑膜炎和正常头颅的分类，所以关注“脑膜病变图像/脑膜炎次诊”“脑膜病变图像/脑膜炎主诊”和“正常头颅MRI”三个目录。这三个目录的直接子目录都是某位患者在某一时刻做的检查的情况，这个检查可能包含多个序列，比如T1WI、T2WI和FLAIR，没有超过这三个之外的序列。

数据预处理的的第一步是将这些散乱的数据整合到该项目根目录的data目录，第一层区分数据集的因素是患病情况，创建两个子文件夹0_normal和1_meningitis（后面可能会拓展数据集中的其他脑部疾病），分别包含脑膜病变图像/脑膜炎次诊”“脑膜病变图像/脑膜炎主诊”中的所有图像，和“正常头颅MRI”中的所有图像。为了存储方便，统一采用.nii.gz格式。第二层区分是MRI序列种类，分别创建1、2、3三个文件夹来整理三种序列的数据。

在原始数据中，每个文件夹（代表某患者的某一次检查）的结尾都是一串数字，这个数字可以对这次检查做唯一标识，即使是同一个人的两次检查，也应该当成两个数据（case），因为其情况是会发生变化的。为了避免繁琐的ID，采用001、002等三位数字重新标识每一个case，1、2、3标识MRI序列种类，例如，脑膜炎的第一个case中检测到T1WI.nii、T2WI.nii、FLAIR.nii，那么应该在/data/1_meningitis/1中添加case_001_1.nii.gz，在/data/1_meningitis/2中添加case_001_2.nii.gz，在/data/1_meningitis/3中添加case_001_3.nii.gz。

为了数据的统一性，实现了对.nii的重采样、归一化和统一尺寸，分别在utils/resample.py、utils/intensity.py和spatial.py。

运行脚本
```bash
python -m scripts.preprocess_data
```

如果实现正确，期望的存储结构如下
```text
./data/processed
    ├── 0_normal/
    │   ├── 1/  # T1WI
    │   ├── 2/  # T2WI
    │   └── 3/  # FLAIR
    ├── 1_meningitis/
    │   ├── 1/
    │   ├── 2/
    │   └── 3/
    └── case_index.json
```

## 生成数据集
下一步是根据预处理之后的数据生成dataset，存储在dataset目录下。存储tensor的文件后缀暂定为.pt，后续可以根据需求，升级为HDF5 / LMDB等。

在utils/dataset.py中，定义了和将MRI图像转化为tensor有关的函数，这些函数在scripts/build_dataset.py脚本被调用。脚本的目标是将每个序列按8:1:1划分为训练集、验证集和测试集。

运行脚本
```bash
python -m scripts.build_dataset
```

如果脚本运行正常，应该会在datasets目录下生成三个子目录seq1_T1、seq2_T2和seq3_FLAIR。每个目录下都有四个文件，分别是train.pt，表示训练数据的张量；val.pt，表示验证数据的张量；test.pt，表示测试数据的张量；json文件，记录训练集和测试集是怎么划分的，保证的项目的可复现性。

期望的结构如下
```text
./datasets/
├── seq1_T1
│   ├── split_seed42.json
│   ├── test.pt
│   ├── train.pt
│   └── val.pt
├── seq2_T2
│   ├── split_seed42.json
│   ├── test.pt
│   ├── train.pt
│   └── val.pt
└── seq3_FLAIR
    ├── split_seed42.json
    ├── test.pt
    ├── train.pt
    └── val.pt

```

## 训练模型
训练模型主要涉及三个文件train.py、configs/train_config.py和models/cnn3d.py。其中，train.py是训练脚本的入口，configs/train_config.py定义了数据路径、输出路径和训练超参数等数据，models/cnn3d.py定义了模型的结构。当然，这个结构不是一成不变的，后续可能添加其他的模型文件，将train.py中的cnn3d替换就行了。

有关超参数（train_config.py）：
```python
NUM_EPOCHS = 100
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 10
```

默认的输入路径是datasets/seq..，输出路径是checkpoints/seq..。train.py有两个参数，--seq表示基于哪一种序列训练，--model表示使用哪种结构的模型。运行脚本
```bash
python train.py --seq 1 --model cnn3d
```
或
```bash
bash  train_command.sh
```
train_command.sh可以自由编辑，例如
```bash
python train.py --seq 1 --model cnn3d
# python tarin.py --seq 2 --model cnn3d
python train.py --seq 3 --model cnn3d
```

训练完成后，应该可以在checkpoints目录下看到输出文件
```text
./checkpoints/
├── seq1_T1
│   ├── model_best.pth
│   └── model_final.pth
├── seq2_T2
│   ├── model_best.pth
│   └── model_final.pth
└── seq3_FLAIR
    ├── model_best.pth
    └── model_final.pth
```

## 当前使用模型
cnn3d.py->Simple3DCNN（引入投票机制）

## 评估模型
评估模型的脚本是eval.py，它评估了模型的下列指标
1. 准确率Accuracy：模型在测试集上预测的准确率
2. 阳性预测值Precision：你说有病的，有多少是真的？
3. 敏感度Recall：病人真的有病，你能不能抓住？
4. F1分数F1-score：精确率Precision和召回率Recall的调和平均数
5. 混淆矩阵Confusion Matrix：TP：正确预测为正例，FN：错误预测为负例，FP错误预测为正例，TN，正确预测为负例

运行脚本
```bash
python eval.py --seq 1
```
或
```bash
bash eval_command.sh
```

结果可以在终端查看。

在现有模型的基础上，新增了三种序列投票判断的机制，以增强模型的能力。投票判断的代码在脚本eval_vote.py中，运行脚本
```bash
python eval_vote.py
```
即可查看模型投票后的各种指标。

## 新机制：K_FOLDS交叉验证
第一次训练的参数：
```python
K_FOLDS = 5
K_FOLDS_VAL_RATIO = 0.1

NUM_EPOCHS = 100
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 10
```

第二次训练的参数：
```python
K_FOLDS = 5
K_FOLDS_VAL_RATIO = 0.15

NUM_EPOCHS = 100
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 20
```