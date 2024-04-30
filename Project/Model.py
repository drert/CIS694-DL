import torch
import os 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision import io
import random 
import torch.nn.functional as F
from torchinfo import summary
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision

model = torchvision.models.detection.maskrcnn_resnet50_fpn()
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x)


