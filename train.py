import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import xml.etree.ElementTree as ET
from PIL import Image
import os
import argparse
import math
import numpy as np
import torchvision.transforms as transforms
from examples.multi_model_fusion import SwintransDetect  # 导入模型
from Dataset import get_voc_loader

class SwinLoss(nn.Module):
    """适配SwintransDetect输出的YOLO风格损失函数"""

    def __init__(self, num_classes=20, reg_max=16, lambda_coord=5.0, lambda_obj=1.0, lambda_cls=1.0):
        super(SwinLoss, self).__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls

        # self.mse_loss = nn.SmoothL1Loss(reduction='sum')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='sum')
        self.ce_loss = nn.CrossEntropyLoss(reduction='sum')

        # DFL损失用于边界框回归
        self.dfl_loss = self._dfl_loss

    def _dfl_loss(self, pred_dist, target_dist):
        """Distribution Focal Loss"""
        target_left = target_dist.long()
        target_right = target_left + 1
        weight_left = target_right.float() - target_dist
        weight_right = target_dist - target_left.float()

        # 确保索引在有效范围内
        target_left = torch.clamp(target_left, 0, self.reg_max - 1)
        target_right = torch.clamp(target_right, 0, self.reg_max - 1)

        loss_left = F.cross_entropy(pred_dist, target_left, reduction='none')
        loss_right = F.cross_entropy(pred_dist, target_right, reduction='none')

        return (loss_left * weight_left + loss_right * weight_right).mean()

    def forward(self, predictions, targets):
        """
        计算损失
        Args:
            predictions: dict包含
                - 'box_pred': [B, L, 4*reg_max] 边界框预测
                - 'cls_pred': [B, L, nc] 分类预测
                - 'obj_pred': [B, L, 1] 置信度预测
            targets: [num_targets, 6] (batch_idx, class, x, y, w, h)
        """
        box_pred = predictions['box_pred']  # [B, L, 4*reg_max]
        cls_pred = predictions['cls_pred']  # [B, L, nc]
        obj_pred = predictions['obj_pred']  # [B, L, 1]

        B, L, _ = box_pred.shape
        H = W = int(L ** 0.5)  # 特征图尺寸
        device = box_pred.device

        # 重塑预测为网格格式 [B, H, W, ...]
        box_pred = box_pred.view(B, H, W, 4 * self.reg_max)
        cls_pred = cls_pred.view(B, H, W, self.num_classes)
        obj_pred = obj_pred.view(B, H, W, 1)

        # 创建目标张量
        target_boxes = torch.zeros(B, H, W, 4, device=device)
        target_obj = torch.zeros(B, H, W, 1, device=device)
        target_cls = torch.zeros(B, H, W, dtype=torch.long, device=device)
        obj_mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)

        # 处理目标标注
        if targets.shape[0] > 0:
            for target in targets:
                batch_idx = int(target[0])
                cls_id = int(target[1])
                x, y, w, h = target[2:6]

                # 转换到网格坐标
                gx = x * W
                gy = y * H
                gi = int(torch.clamp(gx.clone().detach(), 0, W - 1))
                gj = int(torch.clamp(gy.clone().detach(), 0, H - 1))

                # 设置目标
                target_boxes[batch_idx, gj, gi, 0] = gx - gi  # x offset
                target_boxes[batch_idx, gj, gi, 1] = gy - gj  # y offset
                target_boxes[batch_idx, gj, gi, 2] = w * W  # w in grid scale
                target_boxes[batch_idx, gj, gi, 3] = h * H  # h in grid scale

                target_obj[batch_idx, gj, gi, 0] = 1.0
                target_cls[batch_idx, gj, gi] = cls_id
                obj_mask[batch_idx, gj, gi] = True

        # 计算损失
        total_loss = 0
        box_loss = torch.tensor(0.0, device=device)
        obj_loss = torch.tensor(0.0, device=device)
        cls_loss = torch.tensor(0.0, device=device)

        # 1. 边界框损失 (使用DFL)
        if obj_mask.sum() > 0:
            # 重塑box_pred为DFL格式 [B, H, W, 4, reg_max]
            box_pred_dfl = box_pred.view(B, H, W, 4, self.reg_max)

            # 获取有目标的预测和目标
            pred_boxes_obj = box_pred_dfl[obj_mask]  # [num_obj, 4, reg_max]
            target_boxes_obj = target_boxes[obj_mask]  # [num_obj, 4]

            # 将目标框坐标转换为DFL目标分布
            target_boxes_obj = torch.clamp(target_boxes_obj, 0, self.reg_max - 1)

            # 计算每个坐标的DFL损失
            for i in range(4):
                pred_dist = pred_boxes_obj[:, i, :]  # [num_obj, reg_max]
                target_dist = target_boxes_obj[:, i]  # [num_obj]
                box_loss += self.dfl_loss(pred_dist, target_dist)

            box_loss = box_loss / 4.0  # 平均4个坐标的损失

        # 2. 置信度损失 (所有位置)
        obj_loss = self.bce_loss(obj_pred.squeeze(-1), target_obj.squeeze(-1))

        # 3. 分类损失 (只对有目标的位置)
        if obj_mask.sum() > 0:
            pred_cls_obj = cls_pred[obj_mask]  # [num_obj, nc]
            target_cls_obj = target_cls[obj_mask]  # [num_obj]
            cls_loss = self.ce_loss(pred_cls_obj, target_cls_obj)

        # 总损失
        total_loss = (self.lambda_coord * box_loss +
                      self.lambda_obj * obj_loss +
                      self.lambda_cls * cls_loss)

        # 返回5个值以匹配调用代码的期望
        conf_obj_loss = obj_loss  # 有目标位置的置信度损失
        conf_noobj_loss = torch.tensor(0.0, device=device)  # 无目标位置的置信度损失

        return total_loss, box_loss, conf_obj_loss, conf_noobj_loss, cls_loss

def adjust_learning_rate(optimizer, base_lr, decay_rate):
    """调整学习率"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr * decay_rate
    return optimizer

def Train(args):
    """主训练函数"""
    cudnn.benchmark = True

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 初始化模型
    net = SwintransDetect(args)
    net.train()
    net.to(device)

    # 优化器设置
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # 数据加载器
    train_loader = get_voc_loader(
        root_dir=args.voc_root,
        image_set='train',
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers
    )

    # 损失函数
    criterion = SwinLoss(
        num_classes=args.class_num,
        reg_max=16,  # 根据您的SwinDetect设置
        lambda_coord=5.0,
        lambda_obj=1.0,
        lambda_cls=1.0
    )

    # 训练循环设置
    train_loader_length = len(train_loader)

    # 创建保存目录
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    # TensorBoard
    writer = SummaryWriter(log_dir='runs/SwintransDetect')

    print(f'''
        Starting training:
            Batch size: {args.batch_size}
            Learning rate: {args.lr}
            Training size: {train_loader_length}
            Image size: {args.img_size}
            Classes: {args.class_num}
        ''')

    iter_num = math.ceil(train_loader_length)

    for epoch in range(args.epochs):
        print(f'Starting epoch {epoch + 1}/{args.epochs}.')

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f'epoch:{epoch + 1}-------lr:{current_lr}')

        epoch_total_loss = 0
        epoch_box_loss = 0
        epoch_obj_loss = 0
        epoch_cls_loss = 0

        for i, (images, targets, img_ids) in enumerate(train_loader):
            # 数据移到设备
            images = images.to(device, non_blocking=True)
            if targets.shape[0] > 0:
                targets = targets.to(device, non_blocking=True)
            else:
                targets = torch.empty(0, 6).to(device)

            # 前向传播
            outputs = net(images)

            # 计算损失
            total_loss, box_loss, conf_obj_loss, conf_noobj_loss, cls_loss = \
                criterion(outputs, targets)

            # 累计损失
            epoch_total_loss += total_loss.cpu().data.item()
            epoch_box_loss += box_loss.cpu().data.item() if isinstance(box_loss, torch.Tensor) else 0
            epoch_obj_loss += (conf_obj_loss.cpu().data.item() + conf_noobj_loss.cpu().data.item()) if isinstance(
                conf_obj_loss, torch.Tensor) else 0
            epoch_cls_loss += cls_loss.cpu().data.item() if isinstance(cls_loss, torch.Tensor) else 0

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
            optimizer.step()

            # 学习率调整
            if epoch == args.stepvalue1:
                optimizer = adjust_learning_rate(optimizer, base_lr=args.lr, decay_rate=args.lr_decay_gamma)
                print('have updated lr!!')

        # Epoch结束
        print(f'Epoch {epoch + 1} finished!!!')
        print(f'---total_loss: {(epoch_total_loss / iter_num):.5f}--')
        print(f'---box_loss: {(epoch_box_loss / iter_num):.5f}---')
        print(f'---obj_loss: {(epoch_obj_loss / iter_num):.5f}---')
        print(f'---cls_loss: {(epoch_cls_loss / iter_num):.5f}---')

        # TensorBoard记录
        writer.add_scalar('epoch_total_loss/train', epoch_total_loss / iter_num, epoch + 1)
        writer.add_scalar('epoch_box_loss/train', epoch_box_loss / iter_num, epoch + 1)
        writer.add_scalar('epoch_obj_loss/train', epoch_obj_loss / iter_num, epoch + 1)
        writer.add_scalar('epoch_cls_loss/train', epoch_cls_loss / iter_num, epoch + 1)

    torch.save(net.state_dict(), args.save_model_dir + 'Swin_Detect.pth')
    writer.close()


