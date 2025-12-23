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
原始数据的路径是/home/tbing/projects/data/brainMRI，该目录下有“脑膜病变图像”“正常头颅MRI”和“MASK”三个子目录。由于首先处理的是脑膜炎和正常头颅的分类，所以关注“脑膜病变图像/脑膜炎次诊”“脑膜病变图像/脑膜炎主诊”和“正常头颅MRI”三个目录。这三个目录的直接子目录都是某位患者在某一时刻做的检查的情况，这个检查可能包含多个序列，比如T1WI、T2WI、FLAIR、DWI和+C，没有超过这五个之外的序列。

数据预处理的的第一步是将这些散乱的数据整合到该项目根目录的data目录，第一层区分数据集的因素是患病情况，创建两个子文件夹0_normal和1_meningitis（后面可能会拓展数据集中的其他脑部疾病），分别包含脑膜病变图像/脑膜炎次诊”“脑膜病变图像/脑膜炎主诊”中的所有图像，和“正常头颅MRI”中的所有图像。为了存储方便，统一采用.nii.gz格式。第二层区分是MRI序列种类，分别创建1、2、3、4、5五个文件夹来整理五种序列的数据。

在原始数据中，每个文件夹（代表某患者的某一次检查）的结尾都是一串数字，这个数字可以对这次检查做唯一标识，即使是同一个人的两次检查，也应该当成两个数据（case），因为其情况是会发生变化的。为了避免繁琐的ID，采用001、002等三位数字重新标识每一个case，1、2、3、4、5标识MRI序列种类，例如，脑膜炎的第一个case中检测到T1WI.nii、T2WI.nii、FLAIR.nii，那么应该在/data/1_meningitis/1中添加case_001_1.nii.gz，在/data/1_meningitis/2中添加case_001_2.nii.gz，在/data/1_meningitis/3中添加case_001_3.nii.gz。

为了数据的统一性，实现了对.nii的重采样和归一化，分别在utils/resample.py和utils/intensity.py。

运行脚本
```bash
python -m scripts.preprocess_data
```

如果实现正确，期望的存储结构如下
```text
data/processed
    ├── 0_normal/
    │   ├── 1/  # T1WI
    │   ├── 2/  # T2WI
    │   ├── 3/  # FLAIR
    │   ├── 4/  # DWI
    │   └── 5/  # +C
    ├── 1_meningitis/
    │   ├── 1/
    │   ├── 2/
    │   ├── 3/
    │   ├── 4/
    │   └── 5/
    └── case_index.json
```

## 生成数据集
下一步是根据预处理之后的数据生成dataset，存储在dataset目录下。存储tensor的文件后缀暂定为.pt，后续可以根据需求，升级为HDF5 / LMDB等。