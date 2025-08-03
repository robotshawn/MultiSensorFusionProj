import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import argparse
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import json

from examples.multi_model_fusion import SwintransDetect
from Dataset import get_voc_loader


class SwintransDetectTester:
    def __init__(self, model_path, args, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.model_path = os.path.join(model_path, "SwintransDetect_1.pth")

        # VOC2007类别名称
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                        'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        # 加载模型
        self.model = self._load_model()

        # 创建保存目录
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'visualizations'), exist_ok=True)

        # 评估指标
        self.reset_metrics()

    def _load_model(self):
        """加载训练好的模型"""
        model = SwintransDetect(self.args)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        print(f"Model loaded from {self.model_path}")
        return model

    def reset_metrics(self):
        """重置评估指标"""
        self.predictions = []
        self.ground_truths = []
        self.all_detections = []
        self.all_ground_truths = []

    def decode_predictions(self, outputs, conf_threshold=0.5, nms_threshold=0.4):
        """
        解码模型输出为边界框
        Args:
            outputs: 模型输出字典
            conf_threshold: 置信度阈值
            nms_threshold: NMS阈值
        Returns:
            detections: 检测结果列表
        """
        box_pred = outputs['box_pred']  # [B, L, 4*reg_max]
        cls_pred = outputs['cls_pred']  # [B, L, nc]
        obj_pred = outputs['obj_pred']  # [B, L, 1]

        B, L, _ = box_pred.shape
        H = W = int(L ** 0.5)

        print(f"Decoding predictions: B={B}, L={L}, H={H}, W={W}")
        print(f"Using conf_threshold={conf_threshold}, nms_threshold={nms_threshold}")

        detections = []

        for b in range(B):
            batch_detections = []

            # 重塑预测
            box_pred_b = box_pred[b].view(H, W, -1)  # [H, W, 4*reg_max]
            cls_pred_b = cls_pred[b].view(H, W, -1)  # [H, W, nc]
            obj_pred_b = obj_pred[b].view(H, W, 1)  # [H, W, 1]

            # 应用sigmoid到置信度
            obj_conf = torch.sigmoid(obj_pred_b).squeeze(-1)  # [H, W]

            # 应用softmax到分类预测
            cls_prob = F.softmax(cls_pred_b, dim=-1)  # [H, W, nc]

            # 获取最大类别概率和索引
            max_cls_prob, cls_indices = torch.max(cls_prob, dim=-1)  # [H, W]

            # 总置信度 = 目标置信度 * 类别概率
            total_conf = obj_conf * max_cls_prob  # [H, W]

            # 找到高置信度的位置
            high_conf_mask = total_conf > conf_threshold
            high_conf_indices = torch.nonzero(high_conf_mask, as_tuple=False)

            print(f"Batch {b}: Found {len(high_conf_indices)} high confidence predictions (>{conf_threshold})")

            if len(high_conf_indices) == 0:
                detections.append(batch_detections)
                continue

            for idx in high_conf_indices:
                j, i = idx[0].item(), idx[1].item()  # 网格坐标

                # 获取置信度和类别
                confidence = total_conf[j, i].item()
                class_id = cls_indices[j, i].item()

                # 解码边界框 (简化版本，您可能需要根据实际DFL实现调整)
                box_params = box_pred_b[j, i]  # [4*reg_max]

                # 简化解码：取每个坐标的期望值
                reg_max = self.args.reg_max if hasattr(self.args, 'reg_max') else 16
                box_params = box_params.view(4, reg_max)

                # 计算期望值
                range_tensor = torch.arange(reg_max, dtype=torch.float32, device=self.device)
                box_decoded = []
                for k in range(4):
                    prob_dist = F.softmax(box_params[k], dim=0)
                    expected_val = torch.sum(prob_dist * range_tensor)
                    box_decoded.append(expected_val.item())

                # 转换为实际坐标
                x_offset, y_offset, w_grid, h_grid = box_decoded

                # 中心点坐标
                x_center = (i + x_offset) / W
                y_center = (j + y_offset) / H

                # 宽高
                width = w_grid / W
                height = h_grid / H

                # 转换为xyxy格式
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2

                # 裁剪到[0,1]
                x1 = max(0, min(1, x1))
                y1 = max(0, min(1, y1))
                x2 = max(0, min(1, x2))
                y2 = max(0, min(1, y2))

                if x2 > x1 and y2 > y1:  # 有效边界框
                    batch_detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': self.classes[class_id]
                    })

            # NMS
            if len(batch_detections) > 0:
                print(f"Before NMS: {len(batch_detections)} detections")
                batch_detections = self.apply_nms(batch_detections, nms_threshold)
                print(f"After NMS: {len(batch_detections)} detections")

            detections.append(batch_detections)

        return detections

    def apply_nms(self, detections, nms_threshold):
        """应用非最大抑制"""
        if len(detections) == 0:
            return detections

        # 按置信度排序
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

        keep = []
        while detections:
            current = detections.pop(0)
            keep.append(current)

            # 移除与当前检测框IoU过高的框
            detections = [
                det for det in detections
                if self.calculate_iou(current['bbox'], det['bbox']) < nms_threshold
                   or current['class_id'] != det['class_id']
            ]

        return keep

    def calculate_iou(self, box1, box2):
        """计算两个边界框的IoU"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # 计算交集
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        # 计算并集
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def visualize_detections(self, image, detections, gt_boxes=None, save_path=None):
        """可视化检测结果"""
        try:
            print(f"Visualizing detections: {len(detections)} detections found")

            # 转换为numpy数组
            if isinstance(image, torch.Tensor):
                # 反归一化
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image = image.cpu().numpy().transpose(1, 2, 0)
                image = std * image + mean
                image = np.clip(image, 0, 1)

            # 创建图像副本
            vis_image = (image * 255).astype(np.uint8)
            vis_image = Image.fromarray(vis_image)
            draw = ImageDraw.Draw(vis_image)

            # 尝试加载字体
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                try:
                    # 尝试其他常见字体
                    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
                except:
                    font = ImageFont.load_default()

            h, w = vis_image.size[1], vis_image.size[0]
            print(f"Image size: {w}x{h}")

            # 绘制真实标注（绿色）
            gt_count = 0
            if gt_boxes is not None and len(gt_boxes) > 0:
                print(f"Drawing {len(gt_boxes)} ground truth boxes")
                for gt in gt_boxes:
                    try:
                        if len(gt) >= 5:  # [batch_idx, class, x, y, w, h]
                            class_id, x_center, y_center, width, height = gt[1:6]

                            # 转换为像素坐标
                            x1 = int((x_center - width / 2) * w)
                            y1 = int((y_center - height / 2) * h)
                            x2 = int((x_center + width / 2) * w)
                            y2 = int((y_center + height / 2) * h)

                            # 确保坐标在图像范围内
                            x1 = max(0, min(w - 1, x1))
                            y1 = max(0, min(h - 1, y1))
                            x2 = max(0, min(w - 1, x2))
                            y2 = max(0, min(h - 1, y2))

                            if x2 > x1 and y2 > y1:
                                # 绘制边界框
                                draw.rectangle([x1, y1, x2, y2], outline='green', width=3)
                                draw.text((x1, max(0, y1 - 20)), f'GT: {self.classes[int(class_id)]}',
                                          fill='green', font=font)
                                gt_count += 1
                    except Exception as e:
                        print(f"Error drawing GT box: {e}")

            # 绘制预测结果（红色）
            det_count = 0
            for det in detections:
                try:
                    x1, y1, x2, y2 = det['bbox']

                    # 转换为像素坐标
                    x1 = int(x1 * w)
                    y1 = int(y1 * h)
                    x2 = int(x2 * w)
                    y2 = int(y2 * h)

                    # 确保坐标在图像范围内
                    x1 = max(0, min(w - 1, x1))
                    y1 = max(0, min(h - 1, y1))
                    x2 = max(0, min(w - 1, x2))
                    y2 = max(0, min(h - 1, y2))

                    if x2 > x1 and y2 > y1:
                        # 绘制边界框
                        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)

                        # 绘制标签和置信度
                        label = f"{det['class_name']}: {det['confidence']:.2f}"
                        draw.text((x1, max(0, y1 - 20)), label, fill='red', font=font)
                        det_count += 1
                except Exception as e:
                    print(f"Error drawing detection box: {e}")

            print(f"Drew {gt_count} GT boxes and {det_count} detection boxes")

            # 保存图像
            if save_path:
                try:
                    # 确保目录存在
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    vis_image.save(save_path, quality=95)
                    print(f"Visualization saved to {save_path}")
                    # 验证文件是否真的保存了
                    if os.path.exists(save_path):
                        file_size = os.path.getsize(save_path)
                        print(f"File saved successfully, size: {file_size} bytes")
                    else:
                        print(f"Error: File was not saved to {save_path}")
                except Exception as e:
                    print(f"Error saving visualization: {e}")

            return vis_image

        except Exception as e:
            print(f"Error in visualize_detections: {e}")
            import traceback
            traceback.print_exc()
            return None

    def evaluate_batch(self, detections, targets, image_ids):
        """评估一个批次的结果"""
        # 确保targets是numpy数组或转换为numpy数组
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()

        # 如果targets是空的，创建一个空数组
        if targets.size == 0:
            targets = np.array([]).reshape(0, 6)  # 假设每个目标有6个元素

        for i, (dets, img_id) in enumerate(zip(detections, image_ids)):
            # 处理真实标注
            batch_gt = []
            if len(targets) > 0:
                for target in targets:
                    # 安全地访问张量元素
                    if target.ndim == 0:  # 0维张量
                        target_batch_idx = target.item()
                        if len(targets.shape) > 1 and targets.shape[1] >= 6:
                            # 如果是多维数组，跳过这个目标
                            continue
                    else:  # 多维张量
                        if len(target) < 6:  # 确保目标至少有6个元素
                            continue
                        target_batch_idx = target[0]
                        if isinstance(target_batch_idx, torch.Tensor):
                            target_batch_idx = target_batch_idx.item()

                    if target_batch_idx == i:  # 属于当前图像
                        # 安全地提取目标信息
                        try:
                            if target.ndim == 1 and len(target) >= 6:
                                batch_gt.append({
                                    'bbox': [
                                        float(target[2]) - float(target[4]) / 2,  # x1
                                        float(target[3]) - float(target[5]) / 2,  # y1
                                        float(target[2]) + float(target[4]) / 2,  # x2
                                        float(target[3]) + float(target[5]) / 2  # y2
                                    ],
                                    'class_id': int(target[1]),
                                    'image_id': img_id
                                })
                        except (IndexError, ValueError) as e:
                            print(f"Warning: Error processing target: {e}")
                            continue

            # 存储预测和真实标注用于后续评估
            for det in dets:
                self.all_detections.append({
                    'image_id': img_id,
                    'class_id': det['class_id'],
                    'confidence': det['confidence'],
                    'bbox': det['bbox']
                })

            for gt in batch_gt:
                self.all_ground_truths.append(gt)

    def calculate_ap(self, class_id, iou_threshold=0.5):
        """计算单个类别的AP"""
        # 获取该类别的检测和真实标注
        class_dets = [d for d in self.all_detections if d['class_id'] == class_id]
        class_gts = [g for g in self.all_ground_truths if g['class_id'] == class_id]

        if len(class_gts) == 0:
            return 0.0, 0, 0

        if len(class_dets) == 0:
            return 0.0, len(class_gts), 0

        # 按置信度排序
        class_dets = sorted(class_dets, key=lambda x: x['confidence'], reverse=True)

        # 标记真实标注是否被匹配
        gt_matched = {i: False for i in range(len(class_gts))}

        tp = []
        fp = []

        for det in class_dets:
            best_iou = 0
            best_gt_idx = -1

            # 找到最佳匹配的真实标注
            for gt_idx, gt in enumerate(class_gts):
                if gt['image_id'] == det['image_id'] and not gt_matched[gt_idx]:
                    iou = self.calculate_iou(det['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

            # 判断是否为真正例
            if best_iou >= iou_threshold and best_gt_idx != -1:
                tp.append(1)
                fp.append(0)
                gt_matched[best_gt_idx] = True
            else:
                tp.append(0)
                fp.append(1)

        # 计算累积精确率和召回率
        tp = np.array(tp)
        fp = np.array(fp)
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / len(class_gts)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

        # 计算AP
        ap = self.calculate_ap_from_pr(recalls, precisions)

        return ap, len(class_gts), len(class_dets)

    def calculate_ap_from_pr(self, recalls, precisions):
        """从PR曲线计算AP"""
        # 添加端点
        recalls = np.concatenate(([0], recalls, [1]))
        precisions = np.concatenate(([0], precisions, [0]))

        # 确保精确率单调递减
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])

        # 计算面积
        indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
        ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])

        return ap

    def calculate_map(self, iou_threshold=0.5):
        """计算mAP"""
        aps = []
        class_stats = {}

        for class_id in range(len(self.classes)):
            ap, num_gt, num_det = self.calculate_ap(class_id, iou_threshold)
            aps.append(ap)
            class_stats[self.classes[class_id]] = {
                'AP': ap,
                'num_gt': num_gt,
                'num_det': num_det
            }

        map_score = np.mean(aps)
        return map_score, class_stats

    def test(self, test_loader, conf_threshold=0.5, nms_threshold=0.4, save_visualizations=True):
        """测试模型"""
        print("Starting evaluation...")
        self.reset_metrics()

        with torch.no_grad():
            for batch_idx, (images, targets, img_ids) in enumerate(test_loader):
                print(f"Processing batch {batch_idx + 1}/{len(test_loader)}")

                # 移到设备
                images = images.to(self.device)
                if isinstance(targets, torch.Tensor) and targets.numel() > 0:
                    targets = targets.to(self.device)

                # 前向传播
                outputs = self.model(images)

                # 解码预测
                detections = self.decode_predictions(outputs, conf_threshold, nms_threshold)

                # 评估
                self.evaluate_batch(detections, targets, img_ids)

                #可视化保存
                if save_visualizations:
                    print(f"Saving visualizations for batch {batch_idx}")
                    for i, (image, dets, img_id) in enumerate(zip(images, detections, img_ids)):
                        print(f"Processing image {i} with {len(dets)} detections, img_id: {img_id}")

                        # 获取对应的真实标注
                        batch_gt = []
                        if isinstance(targets, torch.Tensor) and targets.numel() > 0:
                            targets_np = targets.cpu().numpy()
                            if targets_np.ndim > 1:
                                gt_for_image = targets_np[targets_np[:, 0] == i]
                                if len(gt_for_image) > 0:
                                    batch_gt = gt_for_image
                                    print(f"Found {len(batch_gt)} ground truth boxes for image {i}")

                        # 创建保存路径
                        vis_dir = os.path.join(self.output_dir, 'visualizations')
                        os.makedirs(vis_dir, exist_ok=True)

                        save_path = os.path.join(
                            vis_dir,
                            f'batch_{batch_idx:03d}_img_{i:02d}_{img_id}.jpg'
                        )
                        print(f"Saving to: {save_path}")

                        # 可视化（即使没有检测结果也保存原图）
                        try:
                            result = self.visualize_detections(image, dets, batch_gt, save_path)
                            if result is None:
                                print(f"Failed to create visualization for image {i}")
                        except Exception as e:
                            print(f"Error in visualization: {e}")
                            import traceback
                            traceback.print_exc()

        # 计算评估指标
        map_50, class_stats = self.calculate_map(iou_threshold=0.5)
        map_75, _ = self.calculate_map(iou_threshold=0.75)

        # 计算不同IoU阈值下的mAP
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
        maps = []
        for iou_th in iou_thresholds:
            map_score, _ = self.calculate_map(iou_threshold=iou_th)
            maps.append(map_score)
        map_50_95 = np.mean(maps)

        # 打印结果
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"mAP@0.5: {map_50:.4f}")
        print(f"mAP@0.75: {map_75:.4f}")
        print(f"mAP@0.5:0.95: {map_50_95:.4f}")
        print("\nPer-class AP@0.5:")
        print("-" * 50)

        total_gt = 0
        total_det = 0
        for class_name, stats in class_stats.items():
            print(f"{class_name:15s}: AP={stats['AP']:.4f}, GT={stats['num_gt']:4d}, Det={stats['num_det']:4d}")
            total_gt += stats['num_gt']
            total_det += stats['num_det']

        print("-" * 50)
        print(f"{'Total':15s}: GT={total_gt:4d}, Det={total_det:4d}")

        # 保存评估结果
        results = {
            'mAP@0.5': map_50,
            'mAP@0.75': map_75,
            'mAP@0.5:0.95': map_50_95,
            'class_stats': class_stats,
            'total_gt': total_gt,
            'total_detections': total_det
        }

        results_path = os.path.join(self.output_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")

        return results


def Test(args):
    # 检查模型文件是否存在
    if not os.path.exists(args.save_model_dir):
        print(f"Error: Model file {args.save_model_dir} not found!")
        return

    # 创建测试器
    tester = SwintransDetectTester(args.save_model_dir, args)

    # 创建测试数据加载器
    test_loader = get_voc_loader(
        root_dir=args.voc_root,
        image_set=args.test_set,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers
    )

    print(f"Testing on {args.test_set} set with {len(test_loader)} batches")

    # 开始测试
    results = tester.test(
        test_loader=test_loader,
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
        save_visualizations=args.save_visualizations
    )

    print("Testing completed!")