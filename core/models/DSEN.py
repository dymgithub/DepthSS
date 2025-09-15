# core/models/DSEN.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule
from mmseg.registry import MODELS


class MLP(nn.Sequential):
    def __init__(self, in_channels, mid_channels, out_channels, norm_cfg):
        super().__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            build_norm_layer(norm_cfg, mid_channels)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            nn.ReLU(inplace=True)
        )


@MODELS.register_module()
class GeometryAwareFusionModule(BaseModule):
    def __init__(self,
                 feat_channels,
                 normal_channels=3,
                 embed_channels=64,
                 num_heads=8,
                 dropout=0.1,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 align_corners=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.align_corners = align_corners

        self.mlp_feat = MLP(feat_channels, feat_channels, embed_channels, norm_cfg)
        self.mlp_normal = MLP(normal_channels, embed_channels // 2, embed_channels, norm_cfg)


        self.self_attention = nn.MultiheadAttention(embed_dim=embed_channels, num_heads=num_heads, dropout=dropout,
                                                    batch_first=True)

        self.norm1 = build_norm_layer(norm_cfg, embed_channels)[1]
        self.norm2 = build_norm_layer(norm_cfg, embed_channels)[1]
        self.ffn = nn.Sequential(
            nn.Conv2d(embed_channels, embed_channels * 2, kernel_size=1, bias=False),
            build_norm_layer(norm_cfg, embed_channels * 2)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels * 2, feat_channels, kernel_size=1, bias=False),
        )

    def forward(self, f_final, normal_map):
        if normal_map.shape[-2:] != f_final.shape[-2:]:
            normal_map = F.interpolate(normal_map,
                                       size=f_final.shape[-2:],
                                       mode='bilinear',
                                       align_corners=self.align_corners)

        feat_embed = self.mlp_feat(f_final)
        normal_embed = self.mlp_normal(normal_map)

        fused_embed = feat_embed + normal_embed
        fused_embed = self.norm1(fused_embed)

        b, c, h, w = fused_embed.shape
        fused_seq = fused_embed.flatten(2).permute(0, 2, 1)

        attn_output, _ = self.self_attention(fused_seq, fused_seq, fused_seq)

        attn_output = attn_output.permute(0, 2, 1).reshape(b, c, h, w)

        fused_embed = fused_embed + attn_output
        fused_embed = self.norm2(fused_embed)

        f_enhanced = self.ffn(fused_embed)

        return f_enhanced