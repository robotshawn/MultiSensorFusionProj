import torch
import argparse
from train import Train
from test import Test

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Set Swin-transformer detector', add_help=False)
    # train
    parser.add_argument('--device', default='cuda', type=str, help='device')
    parser.add_argument('--pretrained_model', default='pretrained_model/swin_tiny_patch4_window7_224.pth', type=str,
                        help='load Pretrained model')
    parser.add_argument('--voc_root', type=str, default='./data/VOCdevkit/VOC2007', help='VOC2007_data')
    parser.add_argument('--stepvalue1', type=int, default=35, help='lr_decay_epoch')
    parser.add_argument('--lr_decay_gamma', default=0.1, type=int, help='learning rate decay')
    parser.add_argument('--lr', default=1e-4, type=int, help='learning rate')
    parser.add_argument('--epochs', default=50, type=int, help='epochs')
    parser.add_argument('--batch_size', default=8, type=int, help='batch_size')
    parser.add_argument('--save_model_dir', default='checkpoint/', type=str, help='save model path')
    parser.add_argument('--embed_dim', default=384, type=int, help='embedding dim')
    parser.add_argument('--dim', default=64, type=int, help='dim')
    parser.add_argument('--encoder_dim', default=[96, 192, 384, 768], type=int, help='dim of each encoder layer')
    parser.add_argument('--img_size', default=352, type=int, help='network input size')
    parser.add_argument('--class_num', default=20, type=int, help='network input size')
    parser.add_argument('--num_workers', default=0, type=int)

    # test
    parser.add_argument('--output_dir', type=str, default='./output', help='测试结果保存目录')
    parser.add_argument('--conf_threshold', type=float, default=0.1, help='置信度阈值')
    parser.add_argument('--nms_threshold', type=float, default=0.4, help='NMS阈值')
    parser.add_argument('--test_set', type=str, default='test', choices=['val', 'test'], help='测试集选择')
    parser.add_argument('--reg_max', type=int, default=16, help='回归最大值')
    parser.add_argument('--save_visualizations', type=bool, default=True, help='是否保存可视化结果')

    args = parser.parse_args()

    # 检查GPU
    if torch.cuda.is_available():
        print(f'CUDA is available! GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('CUDA is not available! Using CPU.')

    # 开始训练
    # Train(args)

    # test
    Test(args)



