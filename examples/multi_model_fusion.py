from ..utils.transformer import Transformer
from ..utils.swin_transformer import swin_transformer



transformer = Transformer(embed_dim=384, depth=6, num_heads=8)
image_backbone = swin_transformer(pretrained=True)
    