import torch
import torch.nn as nn
import timm

class SwinSimpleDetector(nn.Module):
    def __init__(self, num_classes=20, backbone_name='swin_tiny_patch4_window7_224'):
        super(SwinSimpleDetector, self).__init__()

        # 初始化 Swin Transformer backbone（不加载分类头）
        self.backbone = timm.create_model(backbone_name, pretrained=False, features_only=True)

        # ===== 加载本地预训练权重 =====
        checkpoint_path = 'pre_train/swin_tiny_patch4_window7_224.pth'
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 预处理 state_dict（有可能是 {'model': ...} 结构）
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # 过滤掉不匹配的参数
        model_dict = self.backbone.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

        # 加载权重
        model_dict.update(filtered_dict)
        self.backbone.load_state_dict(model_dict)
        print(f"成功加载本地预训练模型：{checkpoint_path}")

        # 检测头设置
        dimen_list = [96, 192, 384, 768]
        in_channels = self.backbone.feature_info[-1]['num_chs']
        self.num_classes = num_classes
        self.det_head = nn.Sequential(
            nn.Conv2d(in_channels, dimen_list[-1], kernel_size=3, padding=1),
            nn.BatchNorm2d(dimen_list[-1]),
            nn.ReLU(),
            nn.Conv2d(dimen_list[-1],dimen_list[-2], kernel_size=3, padding=1),
            nn.BatchNorm2d(dimen_list[-2]),
            nn.ReLU(),
            nn.Conv2d(dimen_list[-2], dimen_list[-3], kernel_size=3, padding=1),
            nn.BatchNorm2d(dimen_list[-3]),
            nn.ReLU(),
            nn.Conv2d(dimen_list[-3], dimen_list[-4], kernel_size=3, padding=1),
            nn.BatchNorm2d(dimen_list[-4]),
            nn.ReLU(),
            nn.Conv2d(dimen_list[-4], 4 + num_classes, kernel_size=1)  # 输出 4 bbox + C 类别
        )

    def forward(self, x):
        feats = self.backbone(x)[-1]  # 特征输出
        feats = feats.permute(0, 3, 1, 2).contiguous()  # 从 [B, H, W, C] -> [B, C, H, W]
        out = self.det_head(feats)
        return out
