# === VOC2007 Swin Transformer 多类别目标检测训练代码 ===
# 注意：此代码适用于 VOC2007 数据集，使用 Swin Transformer 做主干特征提取，
# 输出为密集预测（每个位置预测多个类别和bbox）

import os
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from model import SwinSimpleDetector

# ==== VOC 类别 ====
VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
)
CLASS_TO_IDX = {cls: i + 1 for i, cls in enumerate(VOC_CLASSES)}  # 从1开始，0为背景
NUM_CLASSES = len(VOC_CLASSES) + 1  # 包含背景类


# VOC 数据集类（修正版）
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

        # 返回字典格式，方便后续处理
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.empty((0, 4)),
            'labels': torch.tensor(labels, dtype=torch.long) if len(labels) > 0 else torch.empty((0,), dtype=torch.long)
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


def create_targets(batch_targets, feature_size, img_size=224):
    """为批次数据创建密集预测目标"""
    batch_size = len(batch_targets)
    feat_h, feat_w = feature_size

    # 计算缩放比例
    scale_x = feat_w / img_size
    scale_y = feat_h / img_size

    # 初始化目标张量
    cls_targets = torch.zeros(batch_size, feat_h, feat_w, dtype=torch.long)  # 0为背景
    bbox_targets = torch.zeros(batch_size, 4, feat_h, feat_w)
    bbox_weights = torch.zeros(batch_size, feat_h, feat_w)

    for batch_idx, target in enumerate(batch_targets):
        boxes = target['boxes']
        labels = target['labels']

        if len(boxes) == 0:
            continue

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box

            # 转换到特征图坐标
            feat_x1 = max(0, min(feat_w - 1, int(x1 * scale_x)))
            feat_y1 = max(0, min(feat_h - 1, int(y1 * scale_y)))
            feat_x2 = max(0, min(feat_w - 1, int(x2 * scale_x)))
            feat_y2 = max(0, min(feat_h - 1, int(y2 * scale_y)))

            # 计算中心点作为正样本
            center_x = (feat_x1 + feat_x2) // 2
            center_y = (feat_y1 + feat_y2) // 2

            if 0 <= center_y < feat_h and 0 <= center_x < feat_w:
                cls_targets[batch_idx, center_y, center_x] = label
                bbox_targets[batch_idx, :, center_y, center_x] = torch.tensor([x1, y1, x2, y2])
                bbox_weights[batch_idx, center_y, center_x] = 1.0

    return cls_targets, bbox_targets, bbox_weights


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """Focal Loss处理类别不平衡"""
    ce_loss = F.cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()


def custom_collate_fn(batch):
    """自定义批处理函数"""
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    return images, list(targets)


# 设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ==== 数据加载（训练集） ====
dataset = VOCDataset(
    image_dir="data/VOCdevkit/VOC2007/JPEGImages",
    ann_dir="data/VOCdevkit/VOC2007/Annotations",
    file_list="data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt",
    transform=transform
)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)

# ==== 模型 ====
model = SwinSimpleDetector(num_classes=NUM_CLASSES).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ==== 训练循环（修正的多目标检测版本） ====
for epoch in range(200):
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_bbox_loss = 0

    for batch_idx, (imgs, targets) in enumerate(dataloader):
        imgs = imgs.to(device)

        # 前向传播
        cls_logits, bbox_reg = model(imgs)  # cls_logits: [B, C, H, W], bbox_reg: [B, 4, H, W]

        # 获取特征图尺寸
        feature_size = (cls_logits.size(2), cls_logits.size(3))

        # 创建训练目标
        cls_targets, bbox_targets, bbox_weights = create_targets(targets, feature_size)
        cls_targets = cls_targets.to(device)
        bbox_targets = bbox_targets.to(device)
        bbox_weights = bbox_weights.to(device)

        # 计算分类损失
        cls_pred = cls_logits.permute(0, 2, 3, 1).reshape(-1, NUM_CLASSES)  # [B*H*W, C]
        cls_target = cls_targets.reshape(-1)  # [B*H*W]
        cls_loss = focal_loss(cls_pred, cls_target)

        # 计算回归损失
        if bbox_weights.sum() > 0:
            # 只对有目标的位置计算回归损失
            mask = bbox_weights > 0
            bbox_pred_masked = bbox_reg.permute(0, 2, 3, 1)[mask]  # [N_pos, 4]
            bbox_target_masked = bbox_targets.permute(0, 2, 3, 1)[mask]  # [N_pos, 4]
            bbox_loss = F.smooth_l1_loss(bbox_pred_masked, bbox_target_masked)
        else:
            bbox_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # 总损失
        loss = cls_loss + 2.0 * bbox_loss

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录损失
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_bbox_loss += bbox_loss.item()

    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_bbox_loss = total_bbox_loss / len(dataloader)

    print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f} (Cls: {avg_cls_loss:.4f}, Bbox: {avg_bbox_loss:.4f})")

# 训练全部完成，保存最终模型
os.makedirs("checkpoints", exist_ok=True)
save_path = "checkpoints/swin_detector_final.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, save_path)
print(f"最终模型已保存到 {save_path}")