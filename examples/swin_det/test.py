import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.ops import nms
from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET
from tqdm import tqdm
import numpy as np
from model import SwinSimpleDetector
import matplotlib.pyplot as plt
from collections import defaultdict

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
)
CLASS_TO_IDX = {cls: i + 1 for i, cls in enumerate(VOC_CLASSES)}  # 从1开始，0为背景
IDX_TO_CLASS = {i + 1: cls for i, cls in enumerate(VOC_CLASSES)}
NUM_CLASSES = len(VOC_CLASSES) + 1  # 包含背景类

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


class VOCDataset(Dataset):
    def __init__(self, image_dir, ann_dir, file_list, img_size=224, transform=None):
        self.image_dir = image_dir
        self.ann_dir = ann_dir
        self.img_size = img_size
        self.transform = transform
        with open(file_list) as f:
            self.ids = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        ann_path = os.path.join(self.ann_dir, f"{image_id}.xml")

        img = Image.open(img_path).convert("RGB")
        original_size = img.size  # (width, height)
        boxes, labels = self.parse_voc_xml(ann_path)

        # 缩放边界框到目标尺寸
        if len(boxes) > 0:
            boxes = self.resize_boxes(boxes, original_size, (self.img_size, self.img_size))

        if self.transform:
            img = self.transform(img)

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.empty((0, 4)),
            'labels': torch.tensor(labels, dtype=torch.long) if len(labels) > 0 else torch.empty((0,),
                                                                                                 dtype=torch.long),
            'image_id': image_id
        }

        return img, target

    def parse_voc_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        labels = []
        for obj in root.findall("object"):
            cls_name = obj.find("name").text.lower().strip()
            if cls_name not in CLASS_TO_IDX:
                continue
            label = CLASS_TO_IDX[cls_name]
            bbox = obj.find("bndbox")
            x1 = float(bbox.find("xmin").text)
            y1 = float(bbox.find("ymin").text)
            x2 = float(bbox.find("xmax").text)
            y2 = float(bbox.find("ymax").text)

            # 确保边界框有效
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(label)
        return boxes, labels

    def resize_boxes(self, boxes, original_size, target_size):
        """将边界框从原始尺寸缩放到目标尺寸"""
        orig_w, orig_h = original_size
        target_w, target_h = target_size

        scale_x = target_w / orig_w
        scale_y = target_h / orig_h

        resized_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = x1 * scale_x
            y1 = y1 * scale_y
            x2 = x2 * scale_x
            y2 = y2 * scale_y
            resized_boxes.append([x1, y1, x2, y2])

        return resized_boxes


def post_process_predictions(cls_logits, bbox_reg, conf_threshold=0.3, nms_threshold=0.5):
    """
    后处理预测结果，提取多个目标

    Args:
        cls_logits: [B, num_classes, H, W]
        bbox_reg: [B, 4, H, W]
        conf_threshold: 置信度阈值
        nms_threshold: NMS阈值

    Returns:
        results: list of dicts, 每个dict包含 'boxes', 'scores', 'labels'
    """
    batch_size = cls_logits.size(0)
    results = []

    for batch_idx in range(batch_size):
        cls_scores = F.softmax(cls_logits[batch_idx], dim=0)  # [num_classes, H, W]
        bboxes = bbox_reg[batch_idx]  # [4, H, W]

        all_boxes = []
        all_scores = []
        all_labels = []

        # 遍历每个类别（跳过背景类0）
        for class_id in range(1, cls_scores.size(0)):
            class_scores = cls_scores[class_id]  # [H, W]

            mask = class_scores > conf_threshold
            if not mask.any():
                continue

            positions = torch.nonzero(mask, as_tuple=False)  # [N, 2]
            scores = class_scores[mask]  # [N]

            boxes = []
            for pos in positions:
                y, x = pos
                box = bboxes[:, y, x]  # [4]
                boxes.append(box)

            if len(boxes) == 0:
                continue

            boxes = torch.stack(boxes)  # [N, 4]
            labels = torch.full((len(boxes),), class_id, dtype=torch.long, device=boxes.device)

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        if len(all_boxes) == 0:
            results.append({
                'boxes': torch.empty((0, 4)),
                'scores': torch.empty((0,)),
                'labels': torch.empty((0,), dtype=torch.long)
            })
            continue

        all_boxes = torch.cat(all_boxes, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # 使用 NMS，确保索引和数据在同一个设备上
        keep_indices = nms(all_boxes.cpu(), all_scores.cpu(), nms_threshold)

        # 统一在 CPU 上进行后续处理，避免设备冲突
        final_boxes = all_boxes.cpu()[keep_indices]
        final_scores = all_scores.cpu()[keep_indices]
        final_labels = all_labels.cpu()[keep_indices]

        results.append({
            'boxes': final_boxes,
            'scores': final_scores,
            'labels': final_labels
        })

    return results



def compute_iou(box1, box2):
    """计算两个边界框的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter_area = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area + 1e-6

    return inter_area / union_area


def compute_ap(precision, recall):
    """计算Average Precision (AP)"""
    # 确保precision和recall是排序的
    indices = np.argsort(recall)
    recall = recall[indices]
    precision = precision[indices]

    # 添加边界点
    recall = np.concatenate(([0], recall, [1]))
    precision = np.concatenate(([0], precision, [0]))

    # 计算precision的单调递减序列
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    # 计算面积
    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])

    return ap


def evaluate_detections_advanced(pred_results, gt_targets, iou_thresholds=[0.5, 0.75], conf_threshold=0.5):
    """
    高级评估函数，计算完整的目标检测评估指标

    Args:
        pred_results: 预测结果列表
        gt_targets: 真实标签列表
        iou_thresholds: IoU阈值列表
        conf_threshold: 置信度阈值

    Returns:
        评估结果字典
    """

    # 为每个类别和IoU阈值收集检测结果
    class_detections = defaultdict(lambda: defaultdict(list))  # class_id -> iou_thresh -> [detections]
    class_ground_truths = defaultdict(list)  # class_id -> [ground_truths]

    # 收集所有检测结果和真实标签
    for pred_result, gt_target in zip(pred_results, gt_targets):
        pred_boxes = pred_result['boxes']
        pred_labels = pred_result['labels']
        pred_scores = pred_result['scores']

        gt_boxes = gt_target['boxes']
        gt_labels = gt_target['labels']

        # 过滤低置信度预测
        high_conf_mask = pred_scores >= conf_threshold
        pred_boxes = pred_boxes[high_conf_mask]
        pred_labels = pred_labels[high_conf_mask]
        pred_scores = pred_scores[high_conf_mask]

        # 按置信度排序
        sorted_indices = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[sorted_indices]
        pred_labels = pred_labels[sorted_indices]
        pred_scores = pred_scores[sorted_indices]

        # 收集预测结果
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            detection = {
                'bbox': box.cpu().numpy(),
                'score': score.item(),
                'matched': {iou_thresh: False for iou_thresh in iou_thresholds}
            }
            class_detections[label.item()][0].append(detection)

        # 收集真实标签
        for box, label in zip(gt_boxes, gt_labels):
            gt = {
                'bbox': box.cpu().numpy(),
                'matched': {iou_thresh: False for iou_thresh in iou_thresholds}
            }
            class_ground_truths[label.item()].append(gt)

    # 为每个类别计算AP
    results = {}
    class_aps = defaultdict(dict)

    for class_id in range(1, NUM_CLASSES):
        class_name = IDX_TO_CLASS.get(class_id, f"class_{class_id}")

        if class_id not in class_detections or class_id not in class_ground_truths:
            for iou_thresh in iou_thresholds:
                class_aps[class_id][iou_thresh] = 0.0
            continue

        detections = class_detections[class_id][0]
        ground_truths = class_ground_truths[class_id]

        if len(detections) == 0 or len(ground_truths) == 0:
            for iou_thresh in iou_thresholds:
                class_aps[class_id][iou_thresh] = 0.0
            continue

        # 为每个IoU阈值计算AP
        for iou_thresh in iou_thresholds:
            # 重置匹配状态
            for det in detections:
                det['matched'][iou_thresh] = False
            for gt in ground_truths:
                gt['matched'][iou_thresh] = False

            # 匹配检测结果和真实标签
            tp = np.zeros(len(detections))
            fp = np.zeros(len(detections))

            for det_idx, detection in enumerate(detections):
                best_iou = 0
                best_gt_idx = -1

                for gt_idx, gt in enumerate(ground_truths):
                    if gt['matched'][iou_thresh]:
                        continue

                    iou = compute_iou(detection['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= iou_thresh:
                    tp[det_idx] = 1
                    ground_truths[best_gt_idx]['matched'][iou_thresh] = True
                    detection['matched'][iou_thresh] = True
                else:
                    fp[det_idx] = 1

            # 计算precision和recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)

            recall = tp_cumsum / len(ground_truths)
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

            # 计算AP
            ap = compute_ap(precision, recall)
            class_aps[class_id][iou_thresh] = ap

    # 计算mAP
    map_results = {}
    for iou_thresh in iou_thresholds:
        valid_aps = [class_aps[class_id][iou_thresh] for class_id in class_aps
                     if class_aps[class_id][iou_thresh] > 0]

        if valid_aps:
            map_results[f'mAP@{iou_thresh}'] = np.mean(valid_aps)
        else:
            map_results[f'mAP@{iou_thresh}'] = 0.0

    # 计算mAP@0.5:0.95 (简化版本，只用0.5和0.75)
    if len(iou_thresholds) > 1:
        all_aps = []
        for class_id in class_aps:
            class_ap_mean = np.mean([class_aps[class_id][iou_thresh] for iou_thresh in iou_thresholds])
            if class_ap_mean > 0:
                all_aps.append(class_ap_mean)

        map_results['mAP@0.5:0.95'] = np.mean(all_aps) if all_aps else 0.0

    # 计算总体统计
    total_detections = sum(len(class_detections[class_id][0]) for class_id in class_detections)
    total_ground_truths = sum(len(class_ground_truths[class_id]) for class_id in class_ground_truths)

    # 计算recall和precision (在IoU=0.5下)
    total_tp = 0
    for class_id in class_detections:
        for detection in class_detections[class_id][0]:
            if detection['matched'][0.5]:
                total_tp += 1

    overall_precision = total_tp / total_detections if total_detections > 0 else 0
    overall_recall = total_tp / total_ground_truths if total_ground_truths > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    results.update({
        'class_aps': dict(class_aps),
        'map_results': map_results,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1': overall_f1,
        'total_detections': total_detections,
        'total_ground_truths': total_ground_truths,
        'total_tp': total_tp
    })

    return results


def visualize_detections(image_tensor, pred_result, gt_target, save_path=None):
    """可视化检测结果"""
    # 转换为PIL图像
    image = transforms.ToPILImage()(image_tensor.cpu()).convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()

    # 绘制GT框（绿色）
    gt_boxes = gt_target['boxes']
    gt_labels = gt_target['labels']

    for box, label in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = [int(x) for x in box]
        # 确保坐标顺序正确
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
        class_name = IDX_TO_CLASS.get(label.item(), f"class_{label.item()}")
        draw.text((x1, max(y1 - 15, 0)), f"GT: {class_name}", fill="green", font=font)

    # 绘制预测框（红色）
    pred_boxes = pred_result['boxes']
    pred_labels = pred_result['labels']
    pred_scores = pred_result['scores']

    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        x1, y1, x2, y2 = [int(x) for x in box]
        # 确保坐标顺序正确
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        class_name = IDX_TO_CLASS.get(label.item(), f"class_{label.item()}")
        draw.text((x1, min(y2 + 5, image.height - 10)),
                  f"Pred: {class_name} {score:.2f}", fill="red", font=font)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image.save(save_path)

    return image



def custom_collate_fn(batch):
    """自定义批处理函数"""
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    return images, list(targets)


# 数据加载
test_dataset = VOCDataset(
    image_dir="data/VOC2007/JPEGImages",
    ann_dir="data/VOC2007/Annotations",
    file_list="data/VOC2007/ImageSets/Main/test.txt",
    transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

# 模型加载
model = SwinSimpleDetector(num_classes=NUM_CLASSES).to(device)
checkpoint = torch.load("checkpoints/swin_detector_final.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("模型加载完成，开始评估和可视化...")

# 评估设置
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

all_pred_results = []
all_gt_targets = []

print("开始测试和保存所有图片...")
with torch.no_grad():
    for i, (images, targets) in enumerate(tqdm(test_loader, desc="Testing")):
        images = images.to(device)

        # 前向传播
        cls_logits, bbox_reg = model(images)

        # 后处理得到检测结果
        pred_results = post_process_predictions(cls_logits, bbox_reg,conf_threshold=0.3, nms_threshold=0.5)

        # 保存结果用于评估
        all_pred_results.extend(pred_results)
        all_gt_targets.extend(targets)

        # 保存所有图片的可视化结果
        image_id = targets[0]['image_id']
        vis_img = visualize_detections(
            images[0], pred_results[0], targets[0],
            save_path=os.path.join(output_dir, f"{image_id}.jpg")
        )

print("评估检测结果...")
# 计算高级评估指标
eval_results = evaluate_detections_advanced(
    all_pred_results, all_gt_targets,
    iou_thresholds=[0.5, 0.75],
    conf_threshold=0.3
)

print(f"\n{'=' * 60}")
print(f"           目标检测评估结果")
print(f"{'=' * 60}")

print(f"\n检测统计:")
print(f"  总预测框数: {eval_results['total_detections']}")
print(f"  总真实框数: {eval_results['total_ground_truths']}")
print(f"  正确匹配数: {eval_results['total_tp']}")

print(f"\n整体性能指标:")
print(f"  精确率 (Precision): {eval_results['overall_precision']:.4f}")
print(f"  召回率 (Recall):    {eval_results['overall_recall']:.4f}")
print(f"  F1分数 (F1-Score):  {eval_results['overall_f1']:.4f}")

print(f"\n平均精度 (mAP):")
for metric_name, value in eval_results['map_results'].items():
    print(f"  {metric_name:15s}: {value:.4f}")

print(f"\n各类别详细结果 (AP@IoU=0.5):")
print(f"{'类别':<12} {'AP@0.5':<8} {'AP@0.75':<8}")
print(f"{'-' * 30}")

class_aps = eval_results['class_aps']
for class_id in range(1, NUM_CLASSES):
    if class_id in class_aps:
        class_name = IDX_TO_CLASS.get(class_id, f"class_{class_id}")
        ap_50 = class_aps[class_id][0.5]
        ap_75 = class_aps[class_id][0.75]
        print(f"{class_name:<12} {ap_50:<8.3f} {ap_75:<8.3f}")

print(f"\n{'=' * 60}")
print(f"所有测试图片的可视化结果已保存到: {output_dir}")
print(f"共保存了 {len(test_loader)} 张图片")
print(f"{'=' * 60}")
print("测试完成！")