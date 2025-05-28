import copy
from typing import Optional, List
import torch
import torch.nn.functional as F
from torch import nn, Tensor

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        """
        tgt: 需要预测的目标 query embedding  负责预测物体  用于建模图像当中的物体信息  在每层decoder层中不断的被refine
             [100, bs, 256]  和 query_embed形状相同  且全设置为0
        memory: [h*w, bs, 256]  Encoder输出  具有全局相关性（增强后）的特征表示
        tgt_mask: None
        memory_mask: None
        tgt_key_padding_mask: None
        memory_key_padding_mask: [bs, h*w]  记录Encoder输出特征图的每个位置是否是被pad的（True无效   False有效）
        pos: [h*w, bs, 256]  encoder输出特征图的位置编码
        query_pos: [100, bs, 256]  query embedding/tgt的位置编码  负责建模物体与物体之间的位置关系  随机初始化的
        tgt_mask、memory_mask、tgt_key_padding_mask是防止作弊的 这里都没有使用
        """
        # 第一个self-attention的目的：找到图像中物体的信息 -> tgt
        # 第一个多头自注意力层：输入qkv都和Encoder无关  都来自于tgt/query embedding
        # 通过第一个self-attention  可以不断建模物体与物体之间的关系  可以知道图像当中哪些位置会存在物体  物体信息->tgt
        # query embedding  +  query_pos
        q = k = self.with_pos_embed(tgt, query_pos)
        # masked multi-head self-attention  计算query embedding的自注意力
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,key_padding_mask=tgt_key_padding_mask)[0]

        # add + norm
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 第二个self-attention的目的：不断增强encoder的输出特征，将物体的信息不断加入encoder的输出特征中去，更好地表征了图像中的各个物体
        # 第二个多头注意力层，也叫Encoder-Decoder self attention：key和value来自Encoder层输出   Query来自Decoder层输入
        # 第二个self-attention 可以建模图像 与 物体之间的关系
        # 根据上一步得到的tgt作为query 不断的去encoder输出的特征图中去问（q和k计算相似度）  问图像当中的物体在哪里呢？
        # 问完之后再将物体的位置信息融合encoder输出的特征图中（和v做运算）  这样我得到的v的特征就有 encoder增强后特征信息 + 物体的位置信息
        # query = query embedding  +  query_pos
        # key = encoder输出特征 + 特征位置编码
        # value = encoder输出特征
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        # ada + norm + Feed Forward + add + norm
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # [100, bs, 256]
        # decoder的输出是第一个self-attention输出特征 + 第二个self-attention输出特征
        # 最终的特征：知道图像中物体与物体之间的关系 + encoder增强后的图像特征 + 图像与物体之间的关系
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        # 复制num_layers=decoder_layer=TransformerDecoderLayer
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers   # 6
        self.norm = norm               # LN
        # 是否返回中间层 默认True  因为DETR默认6个Decoder都会返回结果，一起加入损失计算的
        # 每一层Decoder都是逐层解析，逐层加强的，所以前面层的解析效果对后面层的解析是有意义的，所以作者把前面5层的输出也加入损失计算
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        """
        tgt: [100, bs, 256] 需要预测的目标query embedding 和 query_embed形状相同  且全设置为0
                            在每层decoder层中不断的被refine，相当于一次次的被coarse-to-fine的过程
        memory: [h*w, bs, 256]  Encoder输出  具有全局相关性（增强后）的特征表示
        tgt_mask: None
        tgt_key_padding_mask: None
        memory_key_padding_mask: [bs, h*w]  记录Encoder输出特征图的每个位置是否是被pad的(True无效   False有效)
        pos: [h*w, bs, 256]                 特征图的位置编码
        query_pos: [100, bs, 256]    query embedding的位置编码  随机初始化的
        """
        output = tgt   # 初始化query embedding  全是0

        intermediate = []  # 用于存放6层decoder的输出结果

        # 遍历6层decoder
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            # 6层结果全部加入intermediate
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        # 默认执行这里
        # 最后把  6x[100,bs,256] -> [6(6层decoder输出),100,bs,256]
        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)   # 不执行
    
class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_decoder_layers=6, 
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()
        """
        d_model: 编码器里面mlp（前馈神经网络  2个linear层）的hidden dim 512
        nhead: 多头注意力头数 8
        num_encoder_layers: encoder的层数 6
        num_decoder_layers: decoder的层数 6
        dim_feedforward: 前馈神经网络的维度 2048
        dropout: 0.1
        activation: 激活函数类型 relu
        normalize_before: 是否使用前置LN
        return_intermediate_dec: 是否返回decoder中间层结果  False
        """
        # 初始化一个小decoder
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        # 创建整个Decoder层  6个decoder层堆叠
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        # 参数初始化
        self._reset_parameters()

        self.d_model = d_model  # 编码器里面mlp的hidden dim 512
        self.nhead = nhead      # 多头注意力头数 8

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, memory, mask, query_embed, pos_embed):
        """
        src: [bs,256,19,26] 图片输入backbone+1x1conv之后的特征图
        mask: [bs, 19, 26]  用于记录特征图中哪些地方是填充的（原图部分值为False，填充部分值为True）
        query_embed: [100, 256]  类似于传统目标检测里面的anchor  这里设置了100个   需要预测的目标
        pos_embed: [bs, 256, 19, 26]  位置编码
        """
        
        bs, L, C = memory.shape

        # pos_embed: [bs, 256, 19, 26]=[bs,C,H,W] -> [494,bs,256]=[HW,bs,C]
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        # query_embed: [100, 256]=[num,C] -> [100,bs,256]=[num,bs,256]
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        # mask: [bs, 19, 26]=[bs,H,W] -> [bs,494]=[bs,HW]
        mask = mask.flatten(1)

        # tgt: [100, bs, 256] 需要预测的目标query embedding 和 query_embed形状相同  且全设置为0
        #                     在每层decoder层中不断的被refine，相当于一次次的被coarse-to-fine的过程
        tgt = torch.zeros_like(query_embed)
       
        # [6, 100, bs, 256]
        # tgt：需要预测的目标 query embeding
        # memory: 输出特征
        # pos: memory的位置编码
        # query_pos: tgt的位置编码
        memory = memory.permute(1, 0, 2)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,pos=pos_embed, query_pos=query_embed)
        # decoder输出 [6, 100, bs, 256] -> [6, bs, 100, 256]
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, C, L**0.5, L**0.5)
    

def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )
    