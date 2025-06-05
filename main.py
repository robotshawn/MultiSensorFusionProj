import argparse
import datetime
import json
import random
import time
from pathlib import Path
import utils.DETR.util.misc as utils
from utils.Decoder import Criterion_Post

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import os
import torch
import argparse
from examples.multi_model_fusion import ImageIRnet

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--device',default='cuda',type=str,help='device')
    parser.add_argument('--pretrained_model', default='swin_tiny_patch4_window7_224.pth', type=str, help='load Pretrained model')
    # parser.add_argument('--lr_decay_gamma', default=0.1, type=int, help='learning rate decay')
    parser.add_argument('--lr', default=1e-4, type=int, help='learning rate')
    parser.add_argument('--epochs', default=300, type=int, help='epochs')
    parser.add_argument('--batch_size', default=10, type=int, help='batch_size')
    parser.add_argument('--save_model_dir', default='checkpoint/', type=str, help='save model path')
    parser.add_argument('--embed_dim', default=384, type=int, help='embedding dim')
    parser.add_argument('--dim', default=64, type=int, help='dim')
    parser.add_argument('--encoder_dim', default=[96,192,384,768], type=int, help='dim of each encoder layer')

    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',help='start epoch')
    parser.add_argument('--eval', action='store_true', default=False)   # False训练  True测试
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,help='gradient clipping max norm')

    parser.add_argument('--dec_layers', default=1, type=int, help="Number of decoding layers in the transformer") #6
    # backbone输入transformer特征的维度
    parser.add_argument('--dim_feedforward', default=2048, type=int,help="Intermediate size of the feedforward layers in the transformer blocks")
    # transformer中隐藏层的维度
    parser.add_argument('--hidden_dim', default=256, type=int,help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")

    # 多头注意力用几头
    parser.add_argument('--nheads', default=8, type=int,help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

     # Loss  是否使用辅助loss  是否使用其他层decoder一起计算loss 默认是不使用
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher 匹配器的损失比重
    # 分类损失权重  1
    parser.add_argument('--set_cost_class', default=1, type=float,help="Class coefficient in the matching cost")
    # L1回归损失权重  5
    parser.add_argument('--set_cost_bbox', default=5, type=float,help="L1 box coefficient in the matching cost")
    # Giou回归损失权重  2
    parser.add_argument('--set_cost_giou', default=2, type=float,help="giou box coefficient in the matching cost")

    # * Loss coefficients  真正的loss比重
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float, help="Relative classification weight of the no-object class")

    # dataset parameters
    # 数据集的类型
    parser.add_argument('--dataset_file', default='coco')
    # 数据集root
    parser.add_argument('--coco_path', type=str, default='../datasets/coco')

    parser.add_argument('--output_dir', default='output',help='path where to save, empty for no saving')
    parser.add_argument('--seed', default=42, type=int)
    
    return parser
    
def main(args):
    device = torch.device(args.device)

    # fix the seed for reproducibility  固定随机数种子
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = ImageIRnet(args)
    model.to(device)
    criterion, postprocessors = Criterion_Post(args)
    
    # 优化器和学习率调整策略
    optimizer = torch.optim.AdamW(lr=args.lr,weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)