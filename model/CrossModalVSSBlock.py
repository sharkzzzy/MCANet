import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Callable

# 从您的模块导入
from your_mamba_module import VSSBlock, SS2D, PMC, APFG, eca_layer

class CrossModalInteraction(nn.Module):
    """
    跨模态交互模块
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 使用类似SS2D的选择性扫描进行跨模态交互
        self.cross_ss2d_s2o = SS2D(d_model=hidden_dim, d_state=8, expand=1)
        self.cross_ss2d_o2s = SS2D(d_model=hidden_dim, d_state=8, expand=1)
        
        # 交互后的增强
        self.enhance_sar = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        self.enhance_opt = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        
    def forward(self, sar_feat, opt_feat):
        # 跨模态SS2D扫描
        # SAR特征指导OPT
        opt_guided = self.cross_ss2d_s2o(torch.cat([opt_feat, sar_feat], dim=-1)[..., :self.hidden_dim])
        # OPT特征指导SAR
        sar_guided = self.cross_ss2d_o2s(torch.cat([sar_feat, opt_feat], dim=-1)[..., :self.hidden_dim])
        
        # 残差连接并增强
        sar_cross = sar_feat + self.enhance_sar(sar_guided)
        opt_cross = opt_feat + self.enhance_opt(opt_guided)
        
        return sar_cross, opt_cross


class AdaptiveFusion(nn.Module):
    """
    自适应融合模块
    """
    def __init__(self, hidden_dim):
        super().__init__()
        
        # 门控机制
        self.gate_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 1),
            nn.Sigmoid()
        )
        
        # 最终融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, sar_feat, opt_feat):
        # (B, H, W, C) -> (B, C, H, W)
        sar_feat = sar_feat.permute(0, 3, 1, 2)
        opt_feat = opt_feat.permute(0, 3, 1, 2)
        
        # 计算门控权重
        concat_feat = torch.cat([sar_feat, opt_feat], dim=1)
        gates = self.gate_conv(concat_feat)
        sar_gate, opt_gate = torch.chunk(gates, 2, dim=1)
        
        # 加权融合
        weighted_sar = sar_feat * sar_gate
        weighted_opt = opt_feat * opt_gate
        
        # 最终融合
        fused = self.fusion_conv(torch.cat([weighted_sar, weighted_opt], dim=1))
        
        # (B, C, H, W) -> (B, H, W, C)
        fused = fused.permute(0, 2, 3, 1)
        
        return fused


class CrossModalVSSBlock(nn.Module):
    """
    跨模态VSSBlock - 复用您的VSSBlock结构
    """
    def __init__(
        self,
        hidden_dim: int,
        drop_path: float = 0.1,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        
        # SAR和OPT各自的VSSBlock
        self.sar_vss_block = VSSBlock(
            hidden_dim=hidden_dim,
            drop_path=drop_path,
            norm_layer=norm_layer,
            attn_drop_rate=attn_drop_rate,
            d_state=d_state,
            **kwargs
        )
        
        self.opt_vss_block = VSSBlock(
            hidden_dim=hidden_dim,
            drop_path=drop_path,
            norm_layer=norm_layer,
            attn_drop_rate=attn_drop_rate,
            d_state=d_state,
            **kwargs
        )
        
        # 跨模态交互层
        self.cross_modal_interaction = CrossModalInteraction(hidden_dim)
        
        # 融合层
        self.fusion = AdaptiveFusion(hidden_dim)
        
    def forward(self, sar_feat, opt_feat):
        # 各自通过VSSBlock处理
        sar_enhanced = self.sar_vss_block(sar_feat)  # (B, H, W, C)
        opt_enhanced = self.opt_vss_block(opt_feat)  # (B, H, W, C)
        
        # 跨模态交互
        sar_cross, opt_cross = self.cross_modal_interaction(sar_enhanced, opt_enhanced)
        
        # 自适应融合
        fused = self.fusion(sar_cross, opt_cross)
        
        return fused
