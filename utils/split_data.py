import os
import glob
import random
from tqdm import tqdm
from shutil import copy
import cv2
import numpy as np

def check_and_fix_label(lbl_path, output_path):
    """检查标签是否为RGB，如果是则转换为单通道"""
    # 读取标签
    img = cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        print(f"无法读取: {lbl_path}")
        copy(lbl_path, output_path)  # 如果无法读取，直接复制
        return
    
    # 如果是RGB图像（3通道）
    if len(img.shape) == 3 and img.shape[2] == 3:
        # BGR到标签的映射（OpenCV使用BGR格式）
        bgr_to_label = {
            (0, 0, 0): 0,           # 背景 - 黑色
            (204, 102, 0): 1,       # 农田 - 蓝色 RGB(0,102,204) → BGR(204,102,0)
            (255, 0, 0): 2,         # 城市 - 红色 RGB(255,0,0) → BGR(0,0,255)
            (255, 255, 0): 3,       # 村庄 - 青色 RGB(0,255,255) → BGR(255,255,0)
            (0, 0, 255): 4,         # 水体 - 蓝色 RGB(0,0,255) → BGR(255,0,0)
            (85, 167, 0): 5,        # 森林 - 绿色 RGB(0,167,85) → BGR(85,167,0)
            (0, 255, 255): 6,       # 道路 - 黄色 RGB(255,255,0) → BGR(0,255,255)
            (153, 102, 153): 7      # 其他 - 紫色
        }
        
        h, w = img.shape[:2]
        label = np.zeros((h, w), dtype=np.uint8)
        
        # 转换每种颜色
        for bgr_color, label_val in bgr_to_label.items():
            mask = np.all(img == np.array(bgr_color), axis=2)
            label[mask] = label_val
        
        # 保存单通道标签
        cv2.imwrite(output_path, label)
    else:
        # 如果已经是单通道，直接复制
        copy(lbl_path, output_path)

#### only split train and test

input_data_path = '../dataset'
input_sar_path = glob.glob(os.path.join(os.path.join(input_data_path, 'sars'), '*.tif'))
input_opt_path = glob.glob(os.path.join(os.path.join(input_data_path, 'opticals'), '*.tif'))
input_lbl_path = glob.glob(os.path.join(os.path.join(input_data_path, 'lbls'), '*.tif'))

print(input_sar_path, input_opt_path, input_lbl_path)
print('共有{}对影像'.format(len(input_opt_path)))

# 先开始随机打乱数据集
test_path = os.path.join(input_data_path, 'test')
if not os.path.exists(test_path):
    os.makedirs(test_path)
test_sar_path = os.path.join(test_path, 'sar')
test_opt_path = os.path.join(test_path, 'opt')
test_lbl_path = os.path.join(test_path, 'lbl')
if not os.path.exists(test_sar_path):
    os.makedirs(test_sar_path)
if not os.path.exists(test_opt_path):
    os.makedirs(test_opt_path)
if not os.path.exists(test_lbl_path):
    os.makedirs(test_lbl_path)

train_path = os.path.join(input_data_path, 'train')
if not os.path.exists(train_path):
    os.makedirs(train_path)
train_sar_path = os.path.join(train_path, 'sar')
train_opt_path = os.path.join(train_path, 'opt')
train_lbl_path = os.path.join(train_path, 'lbl')
if not os.path.exists(train_sar_path):
    os.makedirs(train_sar_path)
if not os.path.exists(train_opt_path):
    os.makedirs(train_opt_path)
if not os.path.exists(train_lbl_path):
    os.makedirs(train_lbl_path)

val_path = os.path.join(input_data_path, 'val')
if not os.path.exists(val_path):
    os.makedirs(val_path)
val_sar_path = os.path.join(val_path, 'sar')
val_opt_path = os.path.join(val_path, 'opt')
val_lbl_path = os.path.join(val_path, 'lbl')
if not os.path.exists(val_sar_path):
    os.makedirs(val_sar_path)
if not os.path.exists(val_opt_path):
    os.makedirs(val_opt_path)
if not os.path.exists(val_lbl_path):
    os.makedirs(val_lbl_path)


spilt_factor = 0.4
random.seed(0)
random.shuffle(input_sar_path)
random.seed(0)
random.shuffle(input_opt_path)
random.seed(0)
random.shuffle(input_lbl_path)

border_idx = int(len(input_sar_path) * (1 - spilt_factor))  # int(len(input_sar_path)-500)#

# train
print('开始随机分割数据集')
print("train set")
for i in tqdm(range(0, border_idx)):
    sar_name = os.path.basename(input_sar_path[i])
    opt_name = os.path.basename(input_opt_path[i])
    lbl_name = os.path.basename(input_lbl_path[i])

    sar_path = os.path.join(train_sar_path, sar_name)
    opt_path = os.path.join(train_opt_path, opt_name)
    lbl_path = os.path.join(train_lbl_path, lbl_name)

    copy(input_sar_path[i], sar_path)
    copy(input_opt_path[i], opt_path)
    # 检查并修复标签
    check_and_fix_label(input_lbl_path[i], lbl_path)

# val
print("val set")
for i in tqdm(range(border_idx, border_idx + int(len(input_sar_path) * 0.2))):
    sar_name = os.path.basename(input_sar_path[i])
    opt_name = os.path.basename(input_opt_path[i])
    lbl_name = os.path.basename(input_lbl_path[i])

    sar_path = os.path.join(val_sar_path, sar_name)
    opt_path = os.path.join(val_opt_path, opt_name)
    lbl_path = os.path.join(val_lbl_path, lbl_name)

    copy(input_sar_path[i], sar_path)
    copy(input_opt_path[i], opt_path)
    # 检查并修复标签
    check_and_fix_label(input_lbl_path[i], lbl_path)


# test
print("test set")
for i in tqdm(range(border_idx + int(len(input_sar_path) * 0.2), len(input_sar_path))):
    sar_name = os.path.basename(input_sar_path[i])
    opt_name = os.path.basename(input_opt_path[i])
    lbl_name = os.path.basename(input_lbl_path[i])

    sar_path = os.path.join(test_sar_path, sar_name)
    opt_path = os.path.join(test_opt_path, opt_name)
    lbl_path = os.path.join(test_lbl_path, lbl_name)

    copy(input_sar_path[i], sar_path)
    copy(input_opt_path[i], opt_path)
    # 检查并修复标签
    check_and_fix_label(input_lbl_path[i], lbl_path)

# 验证结果
print("\n验证数据集划分结果...")
all_classes = set()
for split in ['train', 'val', 'test']:
    split_path = os.path.join(input_data_path, split, 'lbl')
    split_classes = set()
    files = glob.glob(os.path.join(split_path, '*.tif'))
    
    for f in files[:10]:  # 检查前10个文件
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            unique_vals = np.unique(img)
            split_classes.update(unique_vals)
            all_classes.update(unique_vals)
    
    print(f"{split}集包含的类别: {sorted(split_classes)}")

print(f"\n所有数据集的类别: {sorted(all_classes)}")
expected_classes = set(range(8))
missing_classes = expected_classes - all_classes
if missing_classes:
    print(f"警告：缺少类别 {sorted(missing_classes)}")
else:
    print("所有8个类别都存在！")
