import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from functools import partial
from einops import rearrange
# 导入您的原始模块
from your_original_modules import (
    SS2D, VSSBlock, VSSLayer, VSSLayer_up, PMC, eca_layer, APFG,
    PatchExpand, FinalPatchExpand_X4, DropPath
)


class CrossModalMSK(nn.Module):
    """
    跨模态MSK融合模块
    """
    def __init__(self, in_channels, out_channels):
        super(CrossModalMSK, self).__init__()
        self.fusion_conv = CrossModalFusionConv(in_channels, out_channels)

    def forward(self, sar_x1, sar_x2, sar_x4, opt_x1, opt_x2, opt_x4):
        # 跨模态融合
        x_fused = self.fusion_conv(sar_x1, sar_x2, sar_x4, opt_x1, opt_x2, opt_x4)
        return x_fused


class CrossModalFusionConv(nn.Module):
    """
    跨模态融合卷积
    """
    def __init__(self, in_channels, out_channels, factor=4.0):
        super(CrossModalFusionConv, self).__init__()
        dim = int(out_channels // factor)
        
        # SAR分支处理
        self.sar_down = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)
        self.sar_conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.sar_conv_5x5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2)
        self.sar_conv_7x7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3)
        
        # OPT分支处理
        self.opt_down = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)
        self.opt_conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.opt_conv_5x5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2)
        self.opt_conv_7x7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3)
        
        # 跨模态注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=4,
            batch_first=True
        )
        
        # 空间注意力（保留原有的）
        self.spatial_attention = SpatialAttentionModule()
        
        # 融合和输出
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        
        self.up = nn.Conv2d(dim, out_channels, kernel_size=1, stride=1)

    def forward(self, sar_x1, sar_x2, sar_x4, opt_x1, opt_x2, opt_x4):
        # SAR特征融合
        sar_fused = torch.cat([sar_x1, sar_x2, sar_x4], dim=1)
        sar_fused = self.sar_down(sar_fused)
        
        sar_3x3 = self.sar_conv_3x3(sar_fused)
        sar_5x5 = self.sar_conv_5x5(sar_fused)
        sar_7x7 = self.sar_conv_7x7(sar_fused)
        sar_multi = sar_3x3 + sar_5x5 + sar_7x7
        
        # OPT特征融合
        opt_fused = torch.cat([opt_x1, opt_x2, opt_x4], dim=1)
        opt_fused = self.opt_down(opt_fused)
        
        opt_3x3 = self.opt_conv_3x3(opt_fused)
        opt_5x5 = self.opt_conv_5x5(opt_fused)
        opt_7x7 = self.opt_conv_7x7(opt_fused)
        opt_multi = opt_3x3 + opt_5x5 + opt_7x7
        
        # 跨模态注意力
        B, C, H, W = sar_multi.shape
        sar_seq = sar_multi.permute(0, 2, 3, 1).reshape(B, H*W, C)
        opt_seq = opt_multi.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        # 双向交叉注意力
        cross_seq, _ = self.cross_attention(sar_seq, opt_seq, opt_seq)
        cross_feat = cross_seq.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        # 融合三个特征
        fused = self.fusion(torch.cat([sar_multi + cross_feat, opt_multi + cross_feat], dim=1))
        
        # 空间注意力
        fused = fused * self.spatial_attention(fused)
        
        # 输出
        x_out = self.up(fused)
        
        return x_out


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class MultiModalVSSM(nn.Module):
    """
    多模态VSSM网络
    """
    def __init__(self, patch_size=4, in_chans=1, num_classes=6, depths=[2, 2, 9, 2], 
                 dims=[96, 192, 384, 768], d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)

        # 双分支ResNet18
        dims = [64, 128, 256, 512]
        self.sar_backbone = timm.create_model(
            "resnet18", 
            features_only=True, 
            output_stride=32,
            out_indices=(1, 2, 3, 4), 
            pretrained=True,
            in_chans=1  # SAR单通道
        )
        self.opt_backbone = timm.create_model(
            "resnet18", 
            features_only=True, 
            output_stride=32,
            out_indices=(1, 2, 3, 4), 
            pretrained=True,
            in_chans=4  # 光学4通道
        )
        
        base_dims = 64
        
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.num_features_up = int(dims[0] * 2)
        self.dims = dims
        self.final_upsample = final_upsample

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 解码器层（保持不变）
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2*int(dims[0]*2**(self.num_layers-1-i_layer)),
            int(dims[0]*2**(self.num_layers-1-i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = nn.Sequential(
                    VSSLayer(
                        dim=int(dims[0] * 2 ** (self.num_layers - 1 - i_layer)),
                        depth=2,
                        d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                        drop=drop_rate, 
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:-1]):sum(depths[:])],
                        norm_layer=norm_layer,
                        downsample=None,
                        use_checkpoint=use_checkpoint),
                    PatchExpand(dim=int(self.embed_dim * 2 ** (self.num_layers-1-i_layer)), 
                               dim_scale=2, norm_layer=norm_layer)
                )
            else:
                layer_up = VSSLayer_up(
                    dim=int(dims[0] * 2 ** (self.num_layers-1-i_layer)),
                    depth=depths[(self.num_layers-1-i_layer)],
                    d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                    drop=drop_rate, 
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:(self.num_layers-1-i_layer)]):sum(depths[:(self.num_layers-1-i_layer) + 1])],
                    norm_layer=norm_layer,
                    upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint,
                    layer=i_layer,
                )
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            self.up = FinalPatchExpand_X4(dim_scale=4, dim=self.embed_dim)
            self.output = nn.Conv2d(in_channels=self.embed_dim, out_channels=self.num_classes,
                                   kernel_size=1, bias=False)

        # 训练时的辅助头
        if self.training:
            self.conv4 = nn.Conv2d(base_dims * 2, num_classes, 1, bias=False)
            self.conv3 = nn.Conv2d(base_dims, num_classes, 1, bias=False)
            self.conv2 = nn.Conv2d(base_dims, num_classes, 1, bias=False)

        # 跨模态MSK模块
        hidden_dim = int(base_dims // 4)
        self.cross_modal_msks = nn.ModuleList([            
            CrossModalMSK(hidden_dim * 7 * 2, base_dims),      # 处理低层特征
            CrossModalMSK(hidden_dim * 7 * 2, base_dims * 2),  # 处理中层特征
            CrossModalMSK(hidden_dim * 7 * 2, base_dims * 4),  # 处理高层特征
        ])
        
        # 特征转换层
        self.sar_transfer = nn.ModuleList([
            nn.Conv2d(base_dims, hidden_dim, 1, bias=False),
            nn.Conv2d(base_dims * 2, hidden_dim * 2, 1, bias=False),
            nn.Conv2d(base_dims * 4, hidden_dim * 4, 1, bias=False),
        ])
        
        self.opt_transfer = nn.ModuleList([
            nn.Conv2d(base_dims, hidden_dim, 1, bias=False),
            nn.Conv2d(base_dims * 2, hidden_dim * 2, 1, bias=False),
            nn.Conv2d(base_dims * 4, hidden_dim * 4, 1, bias=False),
        ])

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_up_features(self, x, x_downsample, h, w):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:                
                x = torch.cat([x, x_downsample[3-inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
            
            if self.training and inx == 1:
                tmp = torch.permute(x, (0,3,1,2))
                h4 = self.conv4(tmp)
            if self.training and inx == 2:
                tmp = torch.permute(x, (0,3,1,2))
                h3 = self.conv3(tmp)
            if self.training and inx == 3:
                tmp = torch.permute(x, (0,3,1,2))
                h2 = self.conv2(tmp)
        
        if self.training:
            ah = [h2, h3, h4]
            x = self.norm_up(x)
            return x, ah
        else:
            x = self.norm_up(x)   
            return x

    def up_x4(self, x, h, w):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)
        x = self.output(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        return x

    def forward_multimodal_features(self, sar_x_downsample, opt_x_downsample):
        """
        跨模态特征处理
        """
        # 最后一层特征直接合并
        x_down_last = (sar_x_downsample[-1] + opt_x_downsample[-1]) / 2.0
        
        # 处理前三层特征
        fused_features = []
        for idx in range(3):
            # 特征转换
            sar_feat = self.sar_transfer[idx](sar_x_downsample[idx].permute(0,3,1,2))
            opt_feat = self.opt_transfer[idx](opt_x_downsample[idx].permute(0,3,1,2))
            
            # 多尺度准备（类似原始代码的forward_downfeatures）
            if idx == 0:  # 第一层
                sar_down_3_2 = F.interpolate(self.sar_transfer[1](sar_x_downsample[1].permute(0,3,1,2)), 
                                           scale_factor=2.0, mode="bilinear", align_corners=True)
                sar_down_4_2 = F.interpolate(self.sar_transfer[2](sar_x_downsample[2].permute(0,3,1,2)), 
                                           scale_factor=4.0, mode="bilinear", align_corners=True)
                
                opt_down_3_2 = F.interpolate(self.opt_transfer[1](opt_x_downsample[1].permute(0,3,1,2)), 
                                           scale_factor=2.0, mode="bilinear", align_corners=True)
                opt_down_4_2 = F.interpolate(self.opt_transfer[2](opt_x_downsample[2].permute(0,3,1,2)), 
                                           scale_factor=4.0, mode="bilinear", align_corners=True)
                
                fused = self.cross_modal_msks[idx](sar_feat, sar_down_3_2, sar_down_4_2,
                                                  opt_feat, opt_down_3_2, opt_down_4_2)
            
            elif idx == 1:  # 第二层
                sar_down_4_3 = F.interpolate(self.sar_transfer[2](sar_x_downsample[2].permute(0,3,1,2)), 
                                           scale_factor=2.0, mode="bilinear", align_corners=True)
                sar_down_2_3 = F.interpolate(self.sar_transfer[0](sar_x_downsample[0].permute(0,3,1,2)), 
                                           scale_factor=0.5, mode="bilinear", align_corners=True)
                
                opt_down_4_3 = F.interpolate(self.opt_transfer[2](opt_x_downsample[2].permute(0,3,1,2)), 
                                           scale_factor=2.0, mode="bilinear", align_corners=True)
                opt_down_2_3 = F.interpolate(self.opt_transfer[0](opt_x_downsample[0].permute(0,3,1,2)), 
                                           scale_factor=0.5, mode="bilinear", align_corners=True)
                
                fused = self.cross_modal_msks[idx](sar_feat, sar_down_2_3, sar_down_4_3,
                                                  opt_feat, opt_down_2_3, opt_down_4_3)
            
            else:  # 第三层
                sar_down_2_4 = F.interpolate(self.sar_transfer[0](sar_x_downsample[0].permute(0,3,1,2)), 
                                           scale_factor=0.25, mode="bilinear", align_corners=True)
                sar_down_3_4 = F.interpolate(self.sar_transfer[1](sar_x_downsample[1].permute(0,3,1,2)), 
                                           scale_factor=0.5, mode="bilinear", align_corners=True)
                
                opt_down_2_4 = F.interpolate(self.opt_transfer[0](opt_x_downsample[0].permute(0,3,1,2)), 
                                           scale_factor=0.25, mode="bilinear", align_corners=True)
                opt_down_3_4 = F.interpolate(self.opt_transfer[1](opt_x_downsample[1].permute(0,3,1,2)), 
                                           scale_factor=0.5, mode="bilinear", align_corners=True)
                
                fused = self.cross_modal_msks[idx](sar_feat, sar_down_3_4, sar_down_2_4,
                                                  opt_feat, opt_down_3_4, opt_down_2_4)
            
            fused = fused.permute(0, 2, 3, 1)  # 转回(B, H, W, C)
            fused_features.append(fused)
        
        # 添加最后一层
        fused_features.append(x_down_last)
        
        return fused_features

    def forward(self, x):
        """
        x: 字典，包含 'sar' 和 'opt' 两个键
        或者元组 (sar_img, opt_img)
        """
        # 处理输入
        if isinstance(x, dict):
            sar_img = x['sar']
            opt_img = x['opt']
        elif isinstance(x, (list, tuple)):
            sar_img, opt_img = x
        else:
            raise ValueError("Input should be a dict with keys 'sar' and 'opt', or a tuple (sar, opt)")
        
        b, _, h, w = sar_img.size()
        
        # 双分支特征提取
        sar_res1, sar_res2, sar_res3, sar_res4 = self.sar_backbone(sar_img)
        opt_res1, opt_res2, opt_res3, opt_res4 = self.opt_backbone(opt_img)
        
        # 转换为序列格式
        sar_res1 = sar_res1.permute(0,2,3,1)
        sar_res2 = sar_res2.permute(0,2,3,1)
        sar_res3 = sar_res3.permute(0,2,3,1)
        sar_res4 = sar_res4.permute(0,2,3,1)
        
        opt_res1 = opt_res1.permute(0,2,3,1)
        opt_res2 = opt_res2.permute(0,2,3,1)
        opt_res3 = opt_res3.permute(0,2,3,1)
        opt_res4 = opt_res4.permute(0,2,3,1)
        
        sar_x_downsample = [sar_res1, sar_res2, sar_res3, sar_res4]
        opt_x_downsample = [opt_res1, opt_res2, opt_res3, opt_res4]
        
        # 跨模态融合
        x_downsample = self.forward_multimodal_features(sar_x_downsample, opt_x_downsample)
        x = x_downsample[-1]  # 最高层特征
        
        # 解码
        if self.training:
            x, ah = self.forward_up_features(x, x_downsample, h, w)
            x = self.up_x4(x, h, w)
            return x, ah
        else:
            x = self.forward_up_features(x, x_downsample, h, w)
            x = self.up_x4(x, h, w)
            return x




if __name__ == "__main__":
    model = MultiModalVSSM(num_classes=6).cuda()
    
    # 测试输入
    sar_input = torch.randn(4, 1, 256, 256).cuda()  # SAR单通道
    opt_input = torch.randn(4, 4, 256, 256).cuda()  # 光学4通道
    
    # 方式1：元组输入
    output = model((sar_input, opt_input))
    print(f"Output shape: {output.shape}")
    
    # 方式2：字典输入
    output = model({'sar': sar_input, 'opt': opt_input})
    print(f"Output shape: {output.shape}")
