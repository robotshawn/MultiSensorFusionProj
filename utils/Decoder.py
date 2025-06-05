import torch.nn as nn
import torch
from einops import rearrange
from DETR.models import build_model

class decoder_module(nn.Module):       
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
    

#使用DETR构建解码器
def DETR_decoder(args):
    model, criterion, postprocessors = build_model(args)
    return model

def Criterion_Post(args):
    model, criterion, postprocessors = build_model(args)
    return criterion, postprocessors
# class Seg_Decoder(nn.Module):
#     def __init__(self, in_dim=384, image_size=352):
#         super(Seg_Decoder, self).__init__()

#         self.size = image_size
#         self.upsample1 = nn.Sequential(
#             nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1),
#             nn.BatchNorm2d(in_dim),
#             nn.ConvTranspose2d(in_dim, 128, kernel_size=4, stride=2, padding=1),  # 44x44 -> 88x88
#             nn.ReLU())
#         self.upsample2 = nn.Sequential(
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 88x88 -> 176x176
#             nn.ReLU())
#         self.upsample3 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1),  # 176x176 -> 352x352
#             nn.ReLU(),
#             nn.Conv2d(16, 1, kernel_size=1, padding=1),
#             nn.Sigmoid())
        
#     def forward(self, x):
#         B, L, C = x.shape
#         x = x.transpose(1, 2).reshape(B, C,  self.size // 8, self.size // 8)
        
#         x_seg = self.upsample1(x)  
#         x_seg = self.upsample2(x_seg)  
#         seg_mask = self.upsample3(x_seg)  

#         return seg_mask

# class Det_Decoder(nn.Module):
#     def __init__(self, in_dim=384, image_size=352, num_classes=1, num_boxes=10):
#         super().__init__()
#         self.size = image_size
#         self.num_classes = num_classes
#         self.num_boxes = num_boxes
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_dim, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(256, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((1, 1))
#         )
#         self.fc = nn.Linear(128, num_boxes * (4 + num_classes))

#     def forward(self, x):
#         B, L, C = x.shape
#         x = x.transpose(1, 2).reshape(B, C,  self.size // 8, self.size // 8)

#         x = self.conv(x).squeeze(-1).squeeze(-1)  # (B, 128)
#         out = self.fc(x).view(B, self.num_boxes, 4 + self.num_classes)

#         pred_boxes = out[..., :4].sigmoid()
#         pred_logits = out[..., 4:]

#         return pred_logits, pred_boxes

