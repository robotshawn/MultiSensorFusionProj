import argparse, sys, os, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from ultralytics import YOLO
import torch

import os
os.environ['WANDB_MODE'] = 'disabled'
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def str2bool(str):
    return True if str.lower() == 'true' else False

def transformer_opt(opt):
    opt = vars(opt)
    if opt['unamp']:
        opt['amp'] = False
    else:
        opt['amp'] = True
    del opt['yaml']
    del opt['weight']
    del opt['info']
    del opt['unamp']
    return opt

def parse_opt():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--yaml', type=str, default='ultralytics/cfg/models/STASPPNet/STASPPNet.yaml', help='model.yaml path')
    parser.add_argument('--weight', type=str, default='weight/swin_tiny_patch4_window7_224.pth', help='pretrained model path')
    parser.add_argument('--cfg', type=str, default='hyp.yaml', help='hyperparameters path')
    parser.add_argument('--data', type=str, default='data/datasets/VOC2007.yaml', help='data yaml path')
    
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train for')
    parser.add_argument('--patience', type=int, default=0, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--unamp', action='store_true', help='Unuse Automatic Mixed Precision (AMP) training')
    parser.add_argument('--batch', type=int, default=16, help='number of images per batch (-1 for AutoBatch)')
    parser.add_argument('--imgsz', type=int, default=512, help='size of input images as integer')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--device', type=str, default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', type=str, default='runs_train', help='save to project/name')
    parser.add_argument('--name', type=str, default='GSconv-ablotion-ReLU_1', help='save to project/name')
    parser.add_argument('--resume', type=str, default='', help='resume training from last checkpoint')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'Adamax', 'NAdam', 'RAdam', 'AdamW', 'RMSProp', 'auto'], default='Adam', help='optimizer (auto -> ultralytics/yolo/engine/trainer.py in build_optimizer funciton.)')
    parser.add_argument('--close_mosaic', type=int, default=0, help='(int) disable mosaic augmentation for final epochs')
    parser.add_argument('--info', action="store_true", help='model info verbose')
    
    parser.add_argument('--save', type=str2bool, default='True', help='save train checkpoints and predict results')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--seed', type=int, default=42, help='Global training seed')
    parser.add_argument('--deterministic', action="store_true", default=True, help='whether to enable deterministic mode')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--fraction', type=float, default=1.0, help='dataset fraction to train on (default is 1.0, all images in train set)')
    parser.add_argument('--profile', action='store_true', help='profile ONNX and TensorRT speeds during training for loggers')
    
    # Segmentation
    parser.add_argument('--overlap_mask', type=str2bool, default='True', help='masks should overlap during training (segment train only)')
    parser.add_argument('--mask_ratio', type=int, default=4, help='mask downsample ratio (segment train only)')

    # Classification
    parser.add_argument('--dropout', type=float, default=0.0, help='use dropout regularization (classify train only)')

    return parser.parse_known_args()[0]

class YOLOV8(YOLO):
    '''
    yaml:model.yaml path
    weigth:pretrained model path
    '''
    def __init__(self, yaml='ultralytics/cfg/models/v8/yolov8s.yaml', weight='', task=None) -> None:
        super().__init__(yaml, task)
        if weight:
            if weight.endswith('.pth'):  # Handle .pth files
                state_dict = torch.load(weight, map_location='cpu')
                # Filter and adapt state_dict to match model (if necessary)
                model_state_dict = self.model.state_dict()
                state_dict = {k: v for k, v in state_dict.items() if
                              k in model_state_dict and v.shape == model_state_dict[k].shape}
                self.model.load_state_dict(state_dict, strict=False)  # Partial loading
            else:
                self.load(weight)  # Use default Ultralytics loader for .pt
        # if weight:
        #     self.load(weight)
        
if __name__ == '__main__':
    opt = parse_opt()
    
    model = YOLOV8(yaml=opt.yaml, weight=opt.weight)
    if opt.info:
        model.info(detailed=True, verbose=True)
        model.profile(opt.imgsz)
        
        print('before fuse...')
        model.info(detailed=False, verbose=True)
        print('after fuse...')
        model.fuse()
    else:
        model.train(**transformer_opt(opt))