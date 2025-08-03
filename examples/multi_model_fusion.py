from utils.transformer import Transformer
from utils.swin_transformer import swin_transformer
from utils.Fusion import decoder_module
import torch.nn as nn
from utils.detec_head import SwinDetect
import torch


# 利用swin-transformer以及自定义transformer提取特征并融合，来识别检测。

class SwintransDetect(nn.Module):
    def __init__(self, args):
        super(SwintransDetect, self).__init__()
        self.transformer = Transformer(embed_dim=384, depth=3, num_heads=6, mlp_ratio=3.)
        self.image_backbone = swin_transformer(pretrained=True, args=args)

        self.mlp32 = nn.Sequential(
            nn.Linear(args.encoder_dim[3], args.encoder_dim[2]),
            nn.GELU(),
            nn.Linear(args.encoder_dim[2], args.encoder_dim[2]), )
        self.mlp16 = nn.Sequential(
            nn.Linear(args.encoder_dim[2], args.dim),
            nn.GELU(),
            nn.Linear(args.dim, args.dim), )
        self.mlp8 = nn.Sequential(
            nn.Linear(args.encoder_dim[1], args.dim),
            nn.GELU(),
            nn.Linear(args.dim, args.dim), )
        self.mlp4 = nn.Sequential(
            nn.Linear(args.encoder_dim[0], args.dim),
            nn.GELU(),
            nn.Linear(args.dim, args.dim), )

        self.norm1 = nn.LayerNorm(args.dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(args.dim, args.embed_dim),
            nn.GELU(),
            nn.Linear(args.embed_dim, args.embed_dim),
        )

        self.fuse_32_16 = decoder_module(dim=args.embed_dim, token_dim=args.dim, img_size=args.img_size, ratio=16,
                                         kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.fuse_16_8 = decoder_module(dim=args.embed_dim, token_dim=args.dim, img_size=args.img_size, ratio=8,
                                        kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.fuse_8_4 = decoder_module(dim=args.embed_dim, token_dim=args.dim, img_size=args.img_size, ratio=4,
                                       kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)

        self.norm = nn.LayerNorm(args.embed_dim * 2)
        self.mlp_s = nn.Sequential(
            nn.Linear(args.embed_dim * 2, args.embed_dim),
            nn.GELU(),
            nn.Linear(args.embed_dim, args.embed_dim),
        )

        self.decoder = SwinDetect(nc=args.class_num, embed_dim=args.embed_dim, img_size=args.img_size)

    def forward(self, rgb_input):
        feature_1_4, feature_1_8, feature_1_16, feature_1_32 = self.image_backbone(rgb_input)
        # ir_feature_1_4, ir_feature_1_8, ir_feature_1_16, ir_feature_1_32 = self.image_backbone(ir_input)

        # 多尺度特征融合
        feature_1_32 = self.mlp32(feature_1_32)
        feature_1_16 = self.mlp16(feature_1_16)
        feature_1_16 = self.fuse_32_16(feature_1_32, feature_1_16)
        feature_1_16 = self.mlp1(self.norm1(feature_1_16))

        feature_1_8 = self.mlp8(feature_1_8)
        feature_1_8 = self.fuse_16_8(feature_1_16, feature_1_8)
        feature_1_8 = self.mlp1(self.norm1(feature_1_8))
        #
        # feature_1_4 = self.mlp4(feature_1_4)
        # feature_1_4 = self.fuse_8_4(feature_1_8, feature_1_4)
        # feature_1_4 = self.mlp1(self.norm1(feature_1_4))

        fusion_feature = self.transformer(feature_1_8)

        # 检测头输出
        outputs = self.decoder(fusion_feature)

        return outputs




