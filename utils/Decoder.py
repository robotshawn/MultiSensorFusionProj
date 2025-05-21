import torch.nn as nn
import torch
from einops import rearrange

class decoder_module(nn.Module):       #单模态Stage 3 和 Stage 4融合
    def __init__(self, dim=384, token_dim=64, img_size=224, ratio=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True):
        super(decoder_module, self).__init__()
        self.project = nn.Linear(token_dim, token_dim * kernel_size[0] * kernel_size[1])
        self.upsample = nn.Fold(output_size=(img_size // ratio,  img_size // ratio), kernel_size=kernel_size, stride=stride, padding=padding)
        self.fuse = fuse
        if self.fuse:
            self.concatFuse = nn.Sequential(
                nn.Linear(token_dim*2, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, token_dim),)
            #self.att = Token_performer(dim=token_dim, in_dim=token_dim, kernel_ratio=0.5)

            # project input feature to 64 dim
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
                nn.Linear(dim, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, token_dim),)
    def forward(self, dec_fea, enc_fea=None):
        B,_,C = dec_fea.shape
        if C == 384:
            dec_fea = self.mlp(self.norm(dec_fea))
        dec_fea = self.project(dec_fea)
        dec_fea = self.upsample(dec_fea.transpose(1, 2))
        B, C, _, _ = dec_fea.shape
        dec_fea = dec_fea.view(B, C, -1).transpose(1, 2)
        if self.fuse:
            dec_fea = self.concatFuse(torch.cat([dec_fea, enc_fea], dim=2))
        return dec_fea
    
# 解码器模块（分割+检测，加入Stage 1和Stage 2）
class Decoder(nn.Module):
    def __init__(self, in_dim=384, image_size=352, num_classes=2):
        super(Decoder, self).__init__()

        self.size = image_size
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 128, kernel_size=4, stride=2, padding=1),  # 22x22 -> 44x44
            nn.ReLU())
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 44x44 -> 88x88
            nn.ReLU())
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 88x88 -> 176x176
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 176x176 -> 352x352
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid())
        
        self.skip_conv2 = nn.Conv2d(192, 128, kernel_size=1)  # Stage 2: 192 -> 128
        
        self.skip_conv1 = nn.Conv2d(96, 64, kernel_size=1)  # Stage 1: 96 -> 64
        
        
        # detect head
        self.detect_head = nn.Sequential(
            nn.Conv2d(in_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, num_classes + 4, kernel_size=1))
        
    def forward(self, x, stage1_features, stage2_features):
        B, L, C = x.shape

        x = x.transpose(1, 2).reshape(B, C,  self.size // 16, self.size // 16)
        stage1_features = stage1_features.transpose(1,2).reshape(B, C,  self.size // 4, self.size // 4)
        stage2_features = stage2_features.transpose(1,2).reshape(B, C,  self.size // 8, self.size // 8)

        x_seg = self.upsample1(x)  # [B, 128, 44, 44]
        # stage2_features = self.skip_conv2(stage2_features)
        # x_seg = x_seg + stage2_features
        x_seg = self.upsample2(x_seg)  # [B, 64, 88, 88]
        # stage1_features = self.skip_conv1(stage1_features)
        # x_seg = x_seg + stage1_features
        seg_mask = self.upsample3(x_seg)  # [B, 1, 352, 352]
        
        detect_out = self.detect_head(x)  # [B, num_classes+4, 22, 22]
        b, c, _, _ = detect_out.shape
        detect_out = detect_out.permute(0,2,3,1)
        detect_out = detect_out.reshape(b,-1,c)
        
        return seg_mask, detect_out
