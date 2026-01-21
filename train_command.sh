# python train.py --seq 1 --model ResNet
# python train.py --seq 2 --model ResNet
# python train.py --seq 3 --model ResNet

# 最新要求是不用DWI和+C
# python train.py --seq 4
# python train.py --seq 5

# K-Fold Cross Validation Training
# python train_kfold.py --seq 1 --model ResNet --fold 1
# python train_kfold.py --seq 1 --model ResNet --fold 2
# python train_kfold.py --seq 1 --model ResNet --fold 3
# python train_kfold.py --seq 1 --model ResNet --fold 4
# python train_kfold.py --seq 1 --model ResNet --fold 5

# python train_kfold.py --seq 2 --model ResNet --fold 1
# python train_kfold.py --seq 2 --model ResNet --fold 2
# python train_kfold.py --seq 2 --model ResNet --fold 3
# python train_kfold.py --seq 2 --model ResNet --fold 4
# python train_kfold.py --seq 2 --model ResNet --fold 5

# python train_kfold.py --seq 3 --model ResNet --fold 1
# python train_kfold.py --seq 3 --model ResNet --fold 2
python train_kfold.py --seq 3 --model ResNet --fold 3
python train_kfold.py --seq 3 --model ResNet --fold 4
python train_kfold.py --seq 3 --model ResNet --fold 5