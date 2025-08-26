import torch
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

from examples.multi_model_fusion import SwintransDetect
from Dataset import get_voc_loader


class ObjectDetectionVisualizer:
    def __init__(self, model_path, args, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.model_path = os.path.join(model_path, "SwintransDetect_test.pth")

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

    def _load_model(self):
        """加载训练好的模型"""
        model = SwintransDetect(self.args)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        print(f"Model loaded from {self.model_path}")
        return model

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

                # 解码边界框
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

    def visualize_detections(self, image, detections, save_path=None):
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
                    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
                except:
                    font = ImageFont.load_default()

            h, w = vis_image.size[1], vis_image.size[0]
            print(f"Image size: {w}x{h}")

            # 绘制预测结果（红色边界框）
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

            print(f"Drew {det_count} detection boxes")

            # 保存图像
            if save_path:
                try:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    vis_image.save(save_path, quality=95)
                    print(f"Visualization saved to {save_path}")

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

    def predict_and_visualize(self, test_loader, conf_threshold=0.5, nms_threshold=0.4):
        """预测并可视化结果"""
        print("Starting prediction and visualization...")

        with torch.no_grad():
            for batch_idx, (images, targets, img_ids) in enumerate(test_loader):
                print(f"Processing batch {batch_idx + 1}/{len(test_loader)}")

                # 移到设备
                images = images.to(self.device)

                # 前向传播
                outputs = self.model(images)

                # 解码预测
                detections = self.decode_predictions(outputs, conf_threshold, nms_threshold)

                # 保存可视化结果
                print(f"Saving visualizations for batch {batch_idx}")
                for i, (image, dets, img_id) in enumerate(zip(images, detections, img_ids)):
                    print(f"Processing image {i} with {len(dets)} detections, img_id: {img_id}")

                    # 创建保存路径
                    vis_dir = os.path.join(self.output_dir, 'visualizations')
                    os.makedirs(vis_dir, exist_ok=True)

                    save_path = os.path.join(
                        vis_dir,
                        f'batch_{batch_idx:03d}_img_{i:02d}_{img_id}.jpg'
                    )
                    print(f"Saving to: {save_path}")

                    # 可视化
                    try:
                        result = self.visualize_detections(image, dets, save_path)
                        if result is None:
                            print(f"Failed to create visualization for image {i}")
                    except Exception as e:
                        print(f"Error in visualization: {e}")
                        import traceback
                        traceback.print_exc()

        print("Prediction and visualization completed!")


def predict_and_save(args):
    """主函数：预测并保存边界框可视化结果"""
    # 检查模型文件是否存在
    if not os.path.exists(args.save_model_dir):
        print(f"Error: Model file {args.save_model_dir} not found!")
        return

    # 创建可视化器
    visualizer = ObjectDetectionVisualizer(args.save_model_dir, args)

    # 创建数据加载器
    test_loader = get_voc_loader(
        root_dir=args.voc_root,
        image_set=args.test_set,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers
    )

    print(f"Processing {args.test_set} set with {len(test_loader)} batches")

    # 开始预测和可视化
    visualizer.predict_and_visualize(
        test_loader=test_loader,
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold
    )

    print("Processing completed! Check the output directory for visualized images.")


# 使用示例
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
    parser.add_argument('--test_set', type=str, default='test', help='测试集名称')
    parser.add_argument('--conf_threshold', type=float, default=0.1, help='置信度阈值')
    parser.add_argument('--nms_threshold', type=float, default=0.4, help='NMS阈值')
    parser.add_argument('--reg_max', type=int, default=16, help='回归最大值')

    parser.add_argument('--device', default='cuda', type=str, help='device')
    parser.add_argument('--pretrained_model', default='pretrained_model/swin_tiny_patch4_window7_224.pth', type=str,help='load Pretrained model')
    parser.add_argument('--voc_root', type=str, default='./data/VOCdevkit/VOC2007', help='VOC2007_data')
    parser.add_argument('--stepvalue1', type=int, default=150, help='lr_decay_epoch')
    parser.add_argument('--lr_decay_gamma', default=0.1, type=int, help='learning rate decay')
    parser.add_argument('--lr', default=1e-4, type=int, help='learning rate')
    parser.add_argument('--epochs', default=250, type=int, help='epochs')
    parser.add_argument('--batch_size', default=8, type=int, help='batch_size')
    parser.add_argument('--save_model_dir', default='checkpoint/', type=str, help='save model path')
    parser.add_argument('--embed_dim', default=384, type=int, help='embedding dim')
    parser.add_argument('--dim', default=64, type=int, help='dim')
    parser.add_argument('--encoder_dim', default=[96, 192, 384, 768], type=int, help='dim of each encoder layer')
    parser.add_argument('--img_size', default=352, type=int, help='network input size')
    parser.add_argument('--class_num', default=20, type=int, help='network input size')
    parser.add_argument('--num_workers', default=0, type=int)

    args = parser.parse_args()

    predict_and_save(args)