import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入数据集和模型
from DataSet import WHU_OPT_SARDataset
from geoseg.models.multimodal_vssm import MultiModalVSSM  # 导入您的模型

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 设置随机数种子保证可复现
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Args:
    def __init__(self) -> None:
        # 基础参数
        self.batch_size = 8  # 由于VSSM模型较大，减小batch size
        self.lr = 0.0001  # 使用更小的学习率
        self.epochs = 100
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_classes = 8
        
        # VSSM特定参数
        self.depths = [2, 2, 9, 2]
        self.dims = [96, 192, 384, 768]
        self.d_state = 16
        self.drop_rate = 0.0
        self.attn_drop_rate = 0.0
        self.drop_path_rate = 0.1
        self.use_checkpoint = False  # 是否使用梯度检查点以节省显存
        
        # 优化器参数
        self.weight_decay = 0.05
        self.warmup_epochs = 5
        self.min_lr = 1e-6
        
        # 保存路径
        self.save_dir = 'weight/multimodal_vssm'
        os.makedirs(self.save_dir, exist_ok=True)


def get_lr_scheduler(optimizer, args):
    """余弦退火学习率调度器"""
    from torch.optim.lr_scheduler import CosineAnnealingLR
    return CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor=1e-3):
    """预热学习率"""
    def lr_lambda(step):
        if step >= warmup_iters:
            return 1
        alpha = float(step) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train():
    args = Args()
    
    # 数据加载
    print("Loading dataset...")
    train_dataset = WHU_OPT_SARDataset(
        class_name='whu-sar-opt',
        root='dataset/train'
    )
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_dataset = WHU_OPT_SARDataset(
        class_name='whu-sar-opt',
        root='dataset/val'
    )
    val_dataloader = DataLoader(
        dataset=val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 模型初始化
    print("Initializing model...")
    model = MultiModalVSSM(
        num_classes=args.num_classes,
        depths=args.depths,
        dims=args.dims,
        d_state=args.d_state,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate,
        use_checkpoint=args.use_checkpoint
    ).to(args.device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
    
    # 损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # 忽略无效标签
    
    # 优化器 - 使用AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # 学习率调度器
    lr_scheduler = get_lr_scheduler(optimizer, args)
    warmup_scheduler = warmup_lr_scheduler(
        optimizer, 
        warmup_iters=args.warmup_epochs * len(train_dataloader)
    )
    
    # 记录训练过程
    train_epochs_loss = []
    valid_epochs_loss = []
    train_epochs_acc = []
    valid_epochs_acc = []
    best_val_acc = 0
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler()
    
    print("Starting training...")
    for epoch in range(args.epochs):
        # ======================= Train =======================
        model.train()
        train_epoch_loss = []
        acc, nums = 0, 0
        train_bar = tqdm(train_dataloader, desc=f'Train Epoch {epoch+1}/{args.epochs}')
        
        for idx, (sar, opt, label) in enumerate(train_bar):
            sar = sar.to(args.device)
            opt = opt.to(args.device)
            label = label.to(args.device).long()
            
            # 前向传播 - 使用混合精度
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                # MultiModalVSSM接受元组输入
                outputs = model((sar, opt))
                
                # 处理多尺度输出（如果有）
                if isinstance(outputs, tuple):
                    main_output = outputs[0]
                    aux_outputs = outputs[1]
                    # 主损失
                    loss = criterion(main_output, label)
                    # 辅助损失
                    for aux_output in aux_outputs:
                        aux_output = nn.functional.interpolate(
                            aux_output, size=label.shape[1:], 
                            mode='bilinear', align_corners=False
                        )
                        loss += 0.4 * criterion(aux_output, label)
                else:
                    loss = criterion(outputs, label)
                    main_output = outputs
            
            # 反向传播
            scaler.scale(loss).backward()
            
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # 更新学习率（预热期）
            if epoch < args.warmup_epochs:
                warmup_scheduler.step()
            
            # 统计
            train_epoch_loss.append(loss.item())
            pred = torch.argmax(main_output, dim=1)
            acc += torch.sum(pred == label).item()
            nums += label.numel()
            
            # 更新进度条
            current_lr = optimizer.param_groups[0]['lr']
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*acc/nums:.2f}%',
                'lr': f'{current_lr:.6f}'
            })
        
        train_epochs_loss.append(np.average(train_epoch_loss))
        train_acc = 100 * acc / nums
        train_epochs_acc.append(train_acc)
        print(f"Train acc = {train_acc:.3f}%, loss = {np.average(train_epoch_loss):.4f}")
        
        # ======================= Validation =======================
        model.eval()
        val_epoch_loss = []
        acc, nums = 0, 0
        val_bar = tqdm(val_dataloader, desc=f'Val Epoch {epoch+1}/{args.epochs}')
        
        with torch.no_grad():
            for idx, (sar, opt, label) in enumerate(val_bar):
                sar = sar.to(args.device)
                opt = opt.to(args.device)
                label = label.to(args.device).long()
                
                with torch.cuda.amp.autocast():
                    outputs = model((sar, opt))
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = criterion(outputs, label)
                
                val_epoch_loss.append(loss.item())
                pred = torch.argmax(outputs, dim=1)
                acc += torch.sum(pred == label).item()
                nums += label.numel()
                
                val_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100*acc/nums:.2f}%'
                })
        
        valid_epochs_loss.append(np.average(val_epoch_loss))
        val_acc = 100 * acc / nums
        valid_epochs_acc.append(val_acc)
        
        # 更新学习率
        if epoch >= args.warmup_epochs:
            lr_scheduler.step()
        
        print(f"Epoch = {epoch+1}, valid acc = {val_acc:.2f}%, loss = {np.average(val_epoch_loss):.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'args': args
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"Best model saved with acc: {best_val_acc:.2f}%")
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': args
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'final_model.pth'))
    
    # ======================= Plot =======================
    x = np.linspace(1, args.epochs, args.epochs, dtype=int)
    
    plt.figure(figsize=(12, 4))
    
    # Loss图
    plt.subplot(121)
    plt.plot(x, train_epochs_loss, '-o', label="train_loss", markersize=3)
    plt.plot(x, valid_epochs_loss, '-o', label="valid_loss", markersize=3)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy图
    plt.subplot(122)
    plt.plot(x, train_epochs_acc, '-o', label="train_OA", markersize=3)
    plt.plot(x, valid_epochs_acc, '-o', label="valid_OA", markersize=3)
    plt.xlabel('Epochs')
    plt.ylabel('OA (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'training_curves.png'))
    plt.show()
    
    # 打印最终结果
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == '__main__':
    train()
