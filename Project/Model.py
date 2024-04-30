import torch
import os 
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision import io
import random 
import torch.nn.functional as F
from torchinfo import summary
from torchvision.models.detection import maskrcnn_resnet50_fpn


if __name__ == "__main__" :
    mrcnn = maskrcnn_resnet50_fpn()
    summary(mrcnn, input_size=(4, 1, 128, 128))