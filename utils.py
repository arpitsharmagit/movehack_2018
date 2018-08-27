import os
import cv2
import glob
import math
import matplotlib.pyplot as plt
import re
import numpy as np
from PIL import Image

from cv2 import VideoWriter, VideoWriter_fourcc
#Comment this if you don't need video processing capability

def im2video(src, fps = 30.0, image_size = (1280,720)):
    print('input= > {}'.format(src))
    fourcc = VideoWriter_fourcc(*"MP4V")
    vid = VideoWriter(os.path.join(os.path.abspath(src), "input.mp4"),fourcc, fps, image_size)
    for file in os.listdir(src):
        print('{}'.format(file))
        path = os.path.join(os.path.abspath(src), file)
        img = cv2.imread(path, 1)
        print(path, end = '\r')
        vid.write(img)

im2video("input")