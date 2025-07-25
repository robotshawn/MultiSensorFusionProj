# === VOC2007 Swin Transformer 多类别目标检测训练代码 ===
# 注意：此代码适用于 VOC2007 数据集，使用 Swin Transformer 做主干特征提取，
# 输出为密集预测（每个位置预测多个类别和bbox）

import os
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
from model import SwinSimpleDetector

# ==== VOC 类别 ====
VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
)
CLASS_TO_IDX = {cls: i for i, cls in enumerate(VOC_CLASSES)}
NUM_CLASSES = len(VOC_CLASSES)

# ==== VOC 数据集类（简化） ====
class VOCDataset(Dataset):
    def __init__(self, image_dir, ann_dir, file_list, transform=None):
        self.image_dir = image_dir
        self.ann_dir = ann_dir
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
        boxes, labels = self.parse_voc_xml(ann_path)

        if self.transform:
            img = self.transform(img)

        # NOTE: 这里只返回一张图像的所有 bbox 和 labels（不进行复杂匹配）
        return img, torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

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
            boxes.append([x1, y1, x2, y2])
            labels.append(label)
        return boxes, labels

# ==== 设置 ====
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
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: x)

# ==== 模型 ====
model = SwinSimpleDetector(num_classes=NUM_CLASSES).to(device)
criterion_reg = nn.SmoothL1Loss()
criterion_cls = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ==== 训练循环（简化，每图只用第一个目标参与训练） ====
for epoch in range(200):
    model.train()
    total_loss = 0
    for batch in dataloader:
        imgs, targets, labels = zip(*batch)  # 每项都是 list
        imgs = torch.stack(imgs).to(device)

        # 这里只使用每图的第一个目标进行训练（简化处理）
        bbox_targets = torch.stack([b[0] for b in targets]).to(device)
        cls_targets = torch.tensor([l[0] for l in labels]).to(device)

        out = model(imgs)  # [B, 4+C, H, W]
        out = out.mean(dim=[2, 3])  # 简化：全局平均池化 [B, 4+C]

        bbox_preds = out[:, :4]
        cls_preds = out[:, 4:]

        loss_bbox = criterion_reg(bbox_preds, bbox_targets)
        loss_cls = criterion_cls(cls_preds, cls_targets)
        loss = loss_bbox + loss_cls

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss / len(dataloader):.4f}")
# 训练全部完成，保存最终模型
os.makedirs("checkpoints", exist_ok=True)
save_path = "checkpoints/swin_detector_final.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, save_path)
print(f"最终模型已保存到 {save_path}")