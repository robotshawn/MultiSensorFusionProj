import time
import cv2
import numpy as np
from PIL import Image
from frcnn import FRCNN


image = Image.open(r'D:\workspace\luhangyilong\swin-faster-rcnn-voc\img\000006.jpg')
frcnn = FRCNN()
r_image = frcnn.detect_image(image)
r_image.show()