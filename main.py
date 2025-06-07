import argparse
import datetime
import json
import random
import time
from pathlib import Path
import utils.DETR.util.misc as utils
from utils.Decoder import Criterion_Post
from utils.DETR.datasets import build_dataset, get_coco_api_from_dataset

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import os
import argparse
from engine import evaluate, train_one_epoch
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
    parser.add_argument('--num_workers', default=0, type=int)

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
    model_without_ddp = model
    criterion, postprocessors = Criterion_Post(args)

    # 打印模型参数量
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    # 优化器和学习率调整策略
    optimizer = torch.optim.AdamW(lr=args.lr,weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

     # 创建训练和验证数据集
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=False)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    
    base_ds = get_coco_api_from_dataset(dataset_val)
    
    output_dir = Path(args.output_dir)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # 训练一个epoch
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()   # 调整学习率
        # 保存模型
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        # 验证
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )

        # 保存日志
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    # ---------------------------------------------- 训练结束 ----------------------------------------------
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)