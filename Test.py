

import cv2
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime

from DataSet import WHU_OPT_SARDataset
from model.multimodal_vssm import MultiModalVSSM  # 导入您的模型

class Args:
    def __init__(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = 8
        self.num_classes = 7  # 改为7类
        
        # VSSM模型参数
        self.depths = [2, 2, 9, 2]
        self.dims = [96, 192, 384, 768]
        self.d_state = 16
        
        # 路径设置
        self.model_path = 'checkpoints/multimodal_vssm/best_model.pth'  # 最佳模型路径
        self.save_dir = 'predict/multimodal_vssm'
        os.makedirs(self.save_dir, exist_ok=True)




color_map = {
    0: [0, 102, 204],  # 类别1对应棕色 farmland
    1: [0, 0, 255],  # 类别2对应红色  city
    2: [0, 255, 255],  # 类别3对应黄色 village
    3: [255, 0, 0],  # 类别4对应蓝色  water
    4: [0, 167, 85],  # 类别5对应绿色   forest
    5: [255, 255, 0],  # 类别6对应靛蓝色  road
    6: [153, 102, 153]  # 类别7对应紫色  others
}


def test():
    args = Args()
    
    # 数据加载
    test_dataset = WHU_OPT_SARDataset(
        class_name='whu-sar-opt',
        root='dataset/test'
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,  # 测试时不打乱
        num_workers=4,
        pin_memory=True
    )

    # 模型加载
    print(f"Loading model from {args.model_path}")
    model = MultiModalVSSM(
        num_classes=args.num_classes,
        depths=args.depths,
        dims=args.dims,
        d_state=args.d_state
    ).to(args.device)
    
    # 加载权重
    checkpoint = torch.load(args.model_path, map_location=args.device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Best validation accuracy: {checkpoint.get('best_val_acc', 'unknown'):.2f}%")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()

    # 评估指标
    total_acc = 0
    total_pixels = 0
    class_correct = np.zeros(args.num_classes)
    class_total = np.zeros(args.num_classes)
    confusion_matrix = np.zeros((args.num_classes, args.num_classes))
    
    print("Starting evaluation...")
    with torch.no_grad():
        for idx, (sar, opt, label) in enumerate(tqdm(test_dataloader, desc="Testing")):
            sar = sar.to(args.device)
            opt = opt.to(args.device)
            label = label.to(args.device).long()
            
            # 前向传播 - 注意输入格式
            outputs = model((sar, opt))
            
            # 如果是训练模式的输出（包含辅助输出）
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # 获取预测结果
            predictions = torch.argmax(outputs, dim=1)
            
            # 计算准确率
            valid_mask = (label != 255)  # 忽略无效标签
            correct = (predictions == label) & valid_mask
            total_acc += correct.sum().item()
            total_pixels += valid_mask.sum().item()
            
            # 计算每类准确率
            for c in range(args.num_classes):
                class_mask = (label == c) & valid_mask
                class_correct[c] += ((predictions == c) & class_mask).sum().item()
                class_total[c] += class_mask.sum().item()
            
            # 更新混淆矩阵
            for i in range(args.num_classes):
                for j in range(args.num_classes):
                    mask = (label == i) & (predictions == j) & valid_mask
                    confusion_matrix[i, j] += mask.sum().item()
            
            # 可视化一些预测结果（可选）
            if idx < 5:  # 保存前5个批次的预测
                save_predictions(sar, opt, label, predictions, idx, args.save_dir)
    
    # 计算最终指标
    overall_acc = 100 * total_acc / total_pixels
    print(f"\nOverall Accuracy (OA): {overall_acc:.3f}%")
    
    # 每类准确率
    print("\nPer-class Accuracy:")
    class_names = ['Background', 'Class1', 'Class2', 'Class3', 'Class4', 'Class5', 'Class6']
    for i in range(args.num_classes):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f"  {class_names[i]}: {acc:.2f}%")
        else:
            print(f"  {class_names[i]}: No samples")
    
    # 计算mIoU
    iou_list = []
    print("\nPer-class IoU:")
    for i in range(args.num_classes):
        intersection = confusion_matrix[i, i]
        union = confusion_matrix[i, :].sum() + confusion_matrix[:, i].sum() - intersection
        if union > 0:
            iou = intersection / union
            iou_list.append(iou)
            print(f"  {class_names[i]}: {iou:.4f}")
    
    mean_iou = np.mean(iou_list) if iou_list else 0
    print(f"\nMean IoU: {mean_iou:.4f}")
    
    # 保存结果
    save_results(overall_acc, class_correct, class_total, confusion_matrix, args)
    
    return overall_acc, mean_iou


def save_predictions(sar, opt, label, predictions, batch_idx, save_dir):
    """保存预测可视化结果"""
    batch_size = sar.shape[0]
    
    for i in range(min(batch_size, 2)):  # 每批保存前2张
        # 转换为numpy
        pred_np = predictions[i].cpu().numpy()
        label_np = label[i].cpu().numpy()
        
        # 应用颜色映射
        pred_colored = apply_color_map(pred_np)
        label_colored = apply_color_map(label_np)
        
        # 保存图像
        save_name = f'batch{batch_idx}_sample{i}'
        cv2.imwrite(os.path.join(save_dir, f'{save_name}_pred.png'), pred_colored)
        cv2.imwrite(os.path.join(save_dir, f'{save_name}_gt.png'), label_colored)
        
        # 也可以保存原始SAR和光学图像
        sar_img = sar[i, 0].cpu().numpy() * 255
        opt_img = opt[i, :3].cpu().numpy().transpose(1, 2, 0) * 255
        cv2.imwrite(os.path.join(save_dir, f'{save_name}_sar.png'), sar_img.astype(np.uint8))
        cv2.imwrite(os.path.join(save_dir, f'{save_name}_opt.png'), opt_img.astype(np.uint8))


def apply_color_map(label):
    """将标签转换为彩色图像"""
    h, w = label.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in color_map.items():
        mask = (label == class_id)
        rgb_image[mask] = color
    
    return rgb_image


def save_results(overall_acc, class_correct, class_total, confusion_matrix, args):
    """保存测试结果到文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(args.save_dir, f'test_results_{timestamp}.txt')
    
    with open(result_file, 'w') as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Test Time: {timestamp}\n\n")
        f.write(f"Overall Accuracy: {overall_acc:.3f}%\n\n")
        
        f.write("Per-class Accuracy:\n")
        for i in range(args.num_classes):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                f.write(f"  Class {i}: {acc:.2f}% ({int(class_correct[i])}/{int(class_total[i])})\n")
        
        f.write("\nConfusion Matrix:\n")
        np.savetxt(f, confusion_matrix, fmt='%d')
    
    print(f"\nResults saved to: {result_file}")


def visualize_single_prediction(model, image_path, args):
    """可视化单张图像的预测结果"""
    # 这里需要根据您的具体需求实现
    pass


if __name__ == '__main__':
    # 测试模型
    overall_acc, mean_iou = test()
    
    # 如果想可视化特定图像
    # args = Args()
    # model = load_model(args)
    # visualize_single_prediction(model, 'path/to/image', args)



if __name__ == '__main__':
    args = Args()
    test()
