import torch.nn as nn
import torch


class decoder_module(nn.Module):
    def __init__(self, dim=384, token_dim=64, img_size=224, ratio=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                 fuse=True):
        super(decoder_module, self).__init__()
        self.project = nn.Linear(token_dim, token_dim * kernel_size[0] * kernel_size[1])
        self.upsample = nn.Fold(output_size=(img_size // ratio, img_size // ratio), kernel_size=kernel_size,
                                stride=stride, padding=padding)
        self.fuse = fuse
        if self.fuse:
            self.concatFuse = nn.Sequential(
                nn.Linear(token_dim * 2, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, token_dim), )
            # self.att = Token_performer(dim=token_dim, in_dim=token_dim, kernel_ratio=0.5)

            # project input feature to 64 dim
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim), )

    def forward(self, dec_fea, enc_fea=None):
        B, _, C = dec_fea.shape
        if C == 384:
            dec_fea = self.mlp(self.norm(dec_fea))
        dec_fea = self.project(dec_fea)
        dec_fea = self.upsample(dec_fea.transpose(1, 2))
        B, C, _, _ = dec_fea.shape
        dec_fea = dec_fea.view(B, C, -1).transpose(1, 2)
        if self.fuse:
            dec_fea = self.concatFuse(torch.cat([dec_fea, enc_fea], dim=2))
        return dec_fea





