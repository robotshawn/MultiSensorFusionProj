from utils.transformer import Transformer
from utils.swin_transformer import swin_transformer
from utils.Decoder import decoder_module, DETR_decoder
import torch.nn as nn
import torch

# 火源识别检测项目。多传感器特征融合，使用可见光图像与红外光图像，利用swin-transformer以及自定义transformer提取特征并融合，来检测定位火源。


class ImageIRnet(nn.Module):
    def __init__(self, args):
        super(ImageIRnet, self).__init__()
        self.transformer = Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)
        self.image_backbone = swin_transformer(pretrained=True, args=args)

        self.mlp_1_4 = nn.Sequential(
                nn.Linear(args.encoder_dim[0]*2, args.encoder_dim[2]),
                nn.GELU(),
                nn.Linear(args.encoder_dim[2], args.encoder_dim[2]),)
        self.mlp_1_8 = nn.Sequential(
                nn.Linear(args.encoder_dim[1]*2, args.encoder_dim[2]),
                nn.GELU(),
                nn.Linear(args.encoder_dim[2], args.encoder_dim[2]),)
        self.mlp_1_16 = nn.Sequential(
                nn.Linear(args.encoder_dim[2]*2, args.encoder_dim[2]),
                nn.GELU(),
                nn.Linear(args.encoder_dim[2], args.encoder_dim[2]),)
        self.mlp_1_32 = nn.Sequential(
                nn.Linear(args.encoder_dim[3]*2, args.encoder_dim[3]),
                nn.GELU(),
                nn.Linear(args.encoder_dim[3], args.encoder_dim[2]),)

        self.mlp32_16 = nn.Sequential(
                nn.Linear(args.encoder_dim[2], args.dim),
                nn.GELU(),
                nn.Linear(args.dim, args.dim),) 
        self.mlp16_8 = nn.Sequential(
                nn.Linear(args.encoder_dim[2], args.dim),
                nn.GELU(),
                nn.Linear(args.dim, args.dim),)
        self.mlp8_4 = nn.Sequential(
                nn.Linear(args.encoder_dim[2], args.dim),
                nn.GELU(),
                nn.Linear(args.dim, args.dim),)
        
        self.mlp8 = nn.Sequential(
                nn.Linear(args.encoder_dim[1], args.dim),
                nn.GELU(),
                nn.Linear(args.dim, args.dim),)
        
        self.norm1 = nn.LayerNorm(args.dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(args.dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )
        self.fuse_32_16 = decoder_module(dim=args.embed_dim, token_dim=args.dim, img_size=args.img_size, ratio=16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.fuse_16_8 = decoder_module(dim=args.embed_dim, token_dim=args.dim, img_size=args.img_size, ratio=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.fuse_8_4 = decoder_module(dim=args.embed_dim, token_dim=args.dim, img_size=args.img_size, ratio=4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        
        self.norm = nn.LayerNorm(args.embed_dim * 2)
        self.mlp_s = nn.Sequential(
            nn.Linear(args.embed_dim * 2, args.embed_dim),
            nn.GELU(),
            nn.Linear(args.embed_dim, args.embed_dim),
        )

        self.decoder = DETR_decoder(args)

    def forward(self, rgb_input, ir_input):

        rgb_feature_1_4, rgb_feature_1_8, rgb_feature_1_16, rgb_feature_1_32 = self.image_backbone(rgb_input)
        ir_feature_1_4, ir_feature_1_8, ir_feature_1_16, ir_feature_1_32 = self.image_backbone(ir_input)

        feature_1_4 = self.transformer(self.mlp_1_4(torch.cat((rgb_feature_1_4,ir_feature_1_4),dim=-1)))
        feature_1_8 = self.transformer(self.mlp_1_8(torch.cat((rgb_feature_1_8,ir_feature_1_8),dim=-1)))
        feature_1_16 = self.transformer(self.mlp_1_16(torch.cat((rgb_feature_1_16,ir_feature_1_16),dim=-1)))
        feature_1_32 = self.transformer(self.mlp_1_32(torch.cat((rgb_feature_1_32,ir_feature_1_32),dim=-1)))

        feature_1_16 = self.mlp32_16(feature_1_16)
        feature_1_16 = self.fuse_32_16(feature_1_32, feature_1_16)
        feature_1_8 = self.mlp16_8(feature_1_8)
        feature_1_8 = self.fuse_16_8(feature_1_16,feature_1_8)
        feature_1_4 = self.mlp8_4(feature_1_4)
        feature_1_4 = self.fuse_8_4(feature_1_8, feature_1_4)

        fusion_feature = self.mlp1(self.norm1(feature_1_4))

        outputs = self.decoder(fusion_feature)

        return outputs

        

        
