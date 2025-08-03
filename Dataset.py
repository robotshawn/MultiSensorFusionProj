import torch
from torch.utils import data
import xml.etree.ElementTree as ET
from PIL import Image
import os
import numpy as np
import torchvision.transforms as transforms

# VOC2007数据集类
class VOC2007Dataset(data.Dataset):
    def __init__(self, root_dir, image_set='train', transform=None, img_size=416):
        """
        VOC2007数据集加载器
        Args:
            root_dir: VOC2007数据集根目录
            image_set: 'train', 'val', 或 'test'
            transform: 图像变换
            img_size: 输入图像尺寸
        """
        self.root_dir = root_dir
        self.image_set = image_set
        self.transform = transform
        self.img_size = img_size

        # VOC2007类别名称
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                        'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        # 读取图像列表
        with open(os.path.join(root_dir, 'ImageSets', 'Main', f'{image_set}.txt'), 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        # 加载图像
        img_path = os.path.join(self.root_dir, 'JPEGImages', f'{img_id}.jpg')
        image = Image.open(img_path).convert('RGB')

        # 加载标注
        ann_path = os.path.join(self.root_dir, 'Annotations', f'{img_id}.xml')
        boxes, labels = self._parse_annotation(ann_path)

        # 图像预处理
        orig_w, orig_h = image.size
        if self.transform:
            image = self.transform(image)

        # 处理标注，转换为YOLO格式
        targets = self._process_targets(boxes, labels, orig_w, orig_h)

        return image, targets, img_id

    def _parse_annotation(self, ann_path):
        """解析XML标注文件"""
        tree = ET.parse(ann_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall('object'):
            name = obj.find('name').text
            if name in self.class_to_idx:
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(self.class_to_idx[name])

        return np.array(boxes), np.array(labels)

    def _process_targets(self, boxes, labels, orig_w, orig_h):
        """处理目标标注，转换为训练格式"""
        if len(boxes) == 0:
            return torch.zeros((0, 6))  # [batch_idx, class, x, y, w, h]

        targets = []
        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box

            # 归一化到[0,1]
            x_center = ((xmin + xmax) / 2.0) / orig_w
            y_center = ((ymin + ymax) / 2.0) / orig_h
            width = (xmax - xmin) / orig_w
            height = (ymax - ymin) / orig_h

            targets.append([0, label, x_center, y_center, width, height])

        return torch.tensor(targets, dtype=torch.float32)


def get_voc_loader(root_dir, image_set, batch_size, img_size=416, num_workers=4):
    """创建VOC2007数据加载器"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = VOC2007Dataset(root_dir, image_set, transform, img_size)

    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(image_set == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )

    return loader


def collate_fn(batch):
    """自定义批处理函数"""
    images, targets, img_ids = zip(*batch)

    # 堆叠图像
    images = torch.stack(images, 0)

    # 处理targets，添加batch索引
    for i, target in enumerate(targets):
        target[:, 0] = i  # 设置batch索引

    targets = torch.cat(targets, 0)

    return images, targets, img_ids
