import torch
import torch.nn as nn
import math


class Conv(nn.Module):
    """标准卷积块: Conv + BatchNorm + SiLU"""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, self.autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    @staticmethod
    def autopad(k, p):
        """自动计算padding"""
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        return p


class DFL(nn.Module):
    """Distribution Focal Loss (DFL) module"""

    def __init__(self, c1=16):
        super().__init__()
        self.c1 = c1
        self.conv = nn.Conv1d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1))

    def forward(self, x):
        b, a, c = x.shape  # batch, anchors, channels
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1)).view(b, a, 4)


class SwinDetect(nn.Module):
    """基于Swin Transformer特征处理的卷积检测头"""

    def __init__(self, nc=80, embed_dim=384, img_size=224, reg_max=16):
        """
        初始化检测头
        Args:
            nc (int): 类别数量
            embed_dim (int): transformer特征维度
            img_size (int): 输入图像大小
            reg_max (int): DFL通道数
        """
        super().__init__()

        self.nc = nc  # 类别数
        self.embed_dim = embed_dim  # transformer特征维度
        self.img_size = img_size  # 输入图像大小
        self.reg_max = reg_max  # DFL通道数
        self.feature_size = img_size // 4  # 特征图尺寸

        # 计算中间通道数
        c2 = max(16, embed_dim // 4, self.reg_max * 4)  # 边界框分支通道数
        c3 = max(embed_dim // 4, min(self.nc, 100))  # 分类分支通道数

        # 输入特征预处理层
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        # 边界框回归分支 (cv2)
        self.cv2 = nn.Sequential(
            Conv(embed_dim, c2, 3),  # 3x3卷积
            Conv(c2, c2, 3),  # 3x3卷积
            nn.Conv2d(c2, 4 * self.reg_max, 1)  # 1x1卷积
        )

        # 分类分支 (cv3)
        self.cv3 = nn.Sequential(
            Conv(embed_dim, c3, 3),  # 3x3卷积
            Conv(c3, c3, 3),  # 3x3卷积
            nn.Conv2d(c3, self.nc, 1)  # 1x1卷积
        )

        # 置信度分支 (objectness)
        self.cv_obj = nn.Sequential(
            Conv(embed_dim, c3, 3),  # 3x3卷积
            Conv(c3, c3 // 2, 3),  # 3x3卷积
            nn.Conv2d(c3 // 2, 1, 1)  # 1x1卷积
        )

        # DFL模块
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        # 初始化偏置
        self.bias_init()

    def forward(self, x):
        """
        前向传播
        Args:
            x (tensor): [B, L, C] - Swin Transformer输出特征
                       其中 L = (img_size//4) * (img_size//4)
        Returns:
            dict: 包含各分支预测结果的字典
                - 'box_pred': [B, L, 4*reg_max] 边界框预测
                - 'cls_pred': [B, L, nc] 分类预测
                - 'obj_pred': [B, L, 1] 置信度预测
        """
        B, L, C = x.shape
        H = W = int(L ** 0.5)  # 计算特征图高宽

        # 序列特征预处理
        x = self.feature_proj(x)  # [B, L, C]

        # 重塑为2D特征图
        x = x.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]

        # 通过卷积分支获得预测
        box_pred = self.cv2(x)  # [B, 4*reg_max, H, W]
        cls_pred = self.cv3(x)  # [B, nc, H, W]
        obj_pred = self.cv_obj(x)  # [B, 1, H, W]

        # 重塑为序列格式输出
        box_pred = box_pred.view(B, 4 * self.reg_max, -1).transpose(1, 2)  # [B, L, 4*reg_max]
        cls_pred = cls_pred.view(B, self.nc, -1).transpose(1, 2)  # [B, L, nc]
        obj_pred = obj_pred.view(B, 1, -1).transpose(1, 2)  # [B, L, 1]

        return {
            'box_pred': box_pred,
            'cls_pred': cls_pred,
            'obj_pred': obj_pred
        }

    def bias_init(self):
        """初始化检测头偏置"""
        # 边界框分支偏置初始化
        conv = self.cv2[-1]  # 最后一层卷积
        if hasattr(conv, 'bias') and conv.bias is not None:
            conv.bias.data[:] = 1.0

        # 分类分支偏置初始化
        conv = self.cv3[-1]  # 最后一层卷积
        if hasattr(conv, 'bias') and conv.bias is not None:
            # 基于类别数和特征图大小的先验概率
            conv.bias.data[:self.nc] = math.log(5 / self.nc / (640 / 4) ** 2)

        # 置信度分支偏置初始化
        conv = self.cv_obj[-1]  # 最后一层卷积
        if hasattr(conv, 'bias') and conv.bias is not None:
            # 基于先验概率初始化 (假设1%的位置有目标)
            conv.bias.data[0] = math.log(0.01 / (1 - 0.01))