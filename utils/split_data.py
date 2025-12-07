# correct_split.py
import os
import glob
import random
from tqdm import tqdm
from shutil import copy
import cv2
import numpy as np
from PIL import Image

def check_and_fix_label(lbl_path, output_path):
    """检查标签是否为RGB，如果是则转换为单通道"""
    # 读取标签
    img = cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        print(f"无法读取: {lbl_path}")
        return False
    
    # 如果是RGB图像（3通道）
    if len(img.shape) == 3 and img.shape[2] == 3:
        # RGB到标签的映射
        rgb_to_label = {
            (0, 0, 0): 0,           # 背景 - 黑色
            (204, 102, 0): 1,       # 农田 - 棕色(BGR: 0,102,204)
            (0, 0, 255): 2,         # 城市 - 红色(BGR: 255,0,0)
            (0, 255, 255): 3,       # 村庄 - 黄色(BGR: 255,255,0)
            (255, 0, 0): 4,         # 水体 - 蓝色(BGR: 0,0,255)
            (85, 167, 0): 5,        # 森林 - 绿色(BGR: 0,167,85)
            (0, 255, 255): 6,       # 道路 - 黄色(BGR: 255,255,0)
            (153, 102, 153): 7      # 其他 - 紫色
        }
        
        h, w = img.shape[:2]
        label = np.zeros((h, w), dtype=np.uint8)
        
        # 转换每种颜色（注意OpenCV是BGR格式）
        for bgr_color, label_val in rgb_to_label.items():
            mask = np.all(img == np.array(bgr_color), axis=2)
            label[mask] = label_val
        
        # 保存单通道标签
        cv2.imwrite(output_path, label)
        return True
    
    # 如果已经是单通道
    elif len(img.shape) == 2:
        # 检查标签范围
        unique_vals = np.unique(img)
        
        # 如果是0-7范围，直接复制
        if max(unique_vals) <= 7:
            copy(lbl_path, output_path)
            return True
        
        # 如果是0,10,20...70范围，需要映射
        elif max(unique_vals) == 70:
            label_map = {0:0, 10:1, 20:2, 30:3, 40:4, 50:5, 60:6, 70:7}
            new_label = np.zeros_like(img)
            for old_val, new_val in label_map.items():
                new_label[img == old_val] = new_val
            cv2.imwrite(output_path, new_label)
            return True
        
        # 如果是其他奇怪的值
        else:
            print(f"未知的标签值: {unique_vals}")
            return False
    
    return False

def main():
    """主函数：正确的数据集划分"""
    
    input_data_path = '../dataset'
    
    # 读取数据路径
    input_sar_path = sorted(glob.glob(os.path.join(input_data_path, 'sars', '*.tif')))
    input_opt_path = sorted(glob.glob(os.path.join(input_data_path, 'opticals', '*.tif')))
    input_lbl_path = sorted(glob.glob(os.path.join(input_data_path, 'lbls', '*.tif')))
    
    print(f'SAR图像数量: {len(input_sar_path)}')
    print(f'光学图像数量: {len(input_opt_path)}')
    print(f'标签数量: {len(input_lbl_path)}')
    
    # 确保数量一致
    assert len(input_sar_path) == len(input_opt_path) == len(input_lbl_path), \
        "SAR、光学图像和标签数量不一致！"
    
    # 确保文件名对应
    for i in range(len(input_sar_path)):
        sar_base = os.path.basename(input_sar_path[i]).split('.')[0]
        opt_base = os.path.basename(input_opt_path[i]).split('.')[0]
        lbl_base = os.path.basename(input_lbl_path[i]).split('.')[0]
        assert sar_base == opt_base == lbl_base, \
            f"文件名不匹配: {sar_base}, {opt_base}, {lbl_base}"
    
    print(f'共有 {len(input_opt_path)} 对影像')
    
    # 创建输出目录
    splits = ['train', 'val', 'test']
    subdirs = ['sar', 'opt', 'lbl']
    
    for split in splits:
        split_path = os.path.join(input_data_path, split)
        os.makedirs(split_path, exist_ok=True)
        for subdir in subdirs:
            os.makedirs(os.path.join(split_path, subdir), exist_ok=True)
    
    # 设置随机种子确保可重复性
    random.seed(42)
    
    # 创建索引列表并打乱
    indices = list(range(len(input_sar_path)))
    random.shuffle(indices)
    
    # 划分比例
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2
    
    # 计算边界
    train_end = int(len(indices) * train_ratio)
    val_end = train_end + int(len(indices) * val_ratio)
    
    # 划分索引
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    print(f'\n数据集划分:')
    print(f'训练集: {len(train_indices)} 对 ({len(train_indices)/len(indices)*100:.1f}%)')
    print(f'验证集: {len(val_indices)} 对 ({len(val_indices)/len(indices)*100:.1f}%)')
    print(f'测试集: {len(test_indices)} 对 ({len(test_indices)/len(indices)*100:.1f}%)')
    
    # 处理每个数据集
    for split_name, split_indices in [('train', train_indices), 
                                      ('val', val_indices), 
                                      ('test', test_indices)]:
        
        print(f'\n处理 {split_name} 集...')
        
        # 统计标签情况
        label_stats = {'rgb': 0, 'single_channel': 0, 'fixed': 0, 'error': 0}
        
        for idx in tqdm(split_indices, desc=f'{split_name}'):
            # 获取文件名
            sar_src = input_sar_path[idx]
            opt_src = input_opt_path[idx]
            lbl_src = input_lbl_path[idx]
            
            base_name = os.path.basename(sar_src)
            
            # 目标路径
            sar_dst = os.path.join(input_data_path, split_name, 'sar', base_name)
            opt_dst = os.path.join(input_data_path, split_name, 'opt', base_name)
            lbl_dst = os.path.join(input_data_path, split_name, 'lbl', base_name)
            
            # 复制SAR和光学图像
            copy(sar_src, sar_dst)
            copy(opt_src, opt_dst)
            
            # 检查并处理标签
            # 先检查标签类型
            img = cv2.imread(lbl_src, cv2.IMREAD_UNCHANGED)
            if len(img.shape) == 3:
                label_stats['rgb'] += 1
                label_stats['fixed'] += 1
            else:
                label_stats['single_channel'] += 1
            
            # 处理标签（可能需要转换）
            success = check_and_fix_label(lbl_src, lbl_dst)
            if not success:
                label_stats['error'] += 1
                print(f"错误处理标签: {base_name}")
        
        print(f'{split_name} 集标签统计:')
        print(f'  RGB标签: {label_stats["rgb"]}')
        print(f'  单通道标签: {label_stats["single_channel"]}')
        print(f'  已修复: {label_stats["fixed"]}')
        print(f'  错误: {label_stats["error"]}')
    
    # 验证最终结果
    print('\n验证最终数据集...')
    for split_name in splits:
        print(f'\n{split_name} 集:')
        lbl_dir = os.path.join(input_data_path, split_name, 'lbl')
        
        # 检查前5个标签
        all_values = set()
        files = sorted(os.listdir(lbl_dir))[:5]
        
        for filename in files:
            lbl_path = os.path.join(lbl_dir, filename)
            img = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
            unique_vals = np.unique(img)
            all_values.update(unique_vals)
            print(f'  {filename}: {unique_vals}')
        
        print(f'  所有标签值: {sorted(all_values)}')
    
    print('\n数据集划分完成！')
    print('\n请确保在训练时使用 num_classes=8')

if __name__ == '__main__':
    main()
