import os
import torch
import argparse
from multi_model_fusion import ImageIRnet

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda',type=str,help='device')
    parser.add_argument('--pretrained_model', default='swin_tiny_patch4_window7_224.pth', type=str, help='load Pretrained model')
    parser.add_argument('--lr_decay_gamma', default=0.1, type=int, help='learning rate decay')
    parser.add_argument('--lr', default=1e-4, type=int, help='learning rate')
    parser.add_argument('--epochs', default=30, type=int, help='epochs')
    parser.add_argument('--batch_size', default=10, type=int, help='batch_size')
    parser.add_argument('--save_model_dir', default='checkpoint/', type=str, help='save model path')
    parser.add_argument('--embed_dim', default=384, type=int, help='embedding dim')
    parser.add_argument('--dim', default=64, type=int, help='dim')
    parser.add_argument('--encoder_dim', default=[96,192,384,768], type=int, help='dim of each encoder layer')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    net = ImageIRnet(args)
    
