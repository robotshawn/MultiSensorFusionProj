import torch
import torch.nn as nn
import timm

class SwinSimpleDetector(nn.Module):
    def __init__(self, num_classes=20, backbone_name='swin_tiny_patch4_window7_224'):
        super(SwinSimpleDetector, self).__init__()

        self.backbone = timm.create_model(backbone_name, pretrained=False, features_only=True)

        checkpoint_path = 'pre_train/swin_tiny_patch4_window7_224.pth'
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        state_dict = checkpoint.get('model', checkpoint)
        model_dict = self.backbone.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(filtered_dict)
        self.backbone.load_state_dict(model_dict)
        print(f"成功加载本地预训练模型：{checkpoint_path}")

        dimen_list = [96, 192, 384, 768]
        in_channels = self.backbone.feature_info[-1]['num_chs']
        self.num_classes = num_classes

        # 共享特征提取部分
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, dimen_list[-1], kernel_size=3, padding=1),
            nn.BatchNorm2d(dimen_list[-1]),
            nn.ReLU(),
            nn.Conv2d(dimen_list[-1], dimen_list[-2], kernel_size=3, padding=1),
            nn.BatchNorm2d(dimen_list[-2]),
            nn.ReLU(),
            nn.Conv2d(dimen_list[-2], dimen_list[-3], kernel_size=3, padding=1),
            nn.BatchNorm2d(dimen_list[-3]),
            nn.ReLU(),
            nn.Conv2d(dimen_list[-3], dimen_list[-4], kernel_size=3, padding=1),
            nn.BatchNorm2d(dimen_list[-4]),
            nn.ReLU(),
        )

        # 分类 head（每像素点对应一个 num_classes 的分类预测）
        self.cls_head = nn.Conv2d(dimen_list[-4], num_classes, kernel_size=1)

        # 回归 head（每像素点预测一个边界框 [x1, y1, x2, y2]）
        self.reg_head = nn.Conv2d(dimen_list[-4], 4, kernel_size=1)

    def forward(self, x):
        feats = self.backbone(x)[-1]  # 输出为 [B, H, W, C]
        feats = feats.permute(0, 3, 1, 2).contiguous()  # 转为 [B, C, H, W]

        feats = self.shared_conv(feats)
        cls_logits = self.cls_head(feats)  # [B, num_classes, H, W]
        bbox_reg = self.reg_head(feats)    # [B, 4, H, W]

        return cls_logits, bbox_reg
