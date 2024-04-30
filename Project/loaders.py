import os
from scipy.io import loadmat
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm # progress bar visualizer
import torch, torchvision

mpii_idx_to_jnt = {0: 'rankl', 1: 'rknee', 2: 'rhip', 5: 'lankl', 4: 'lknee', 3: 'lhip',
                   6: 'pelvis', 7: 'thorax', 8: 'upper_neck', 11: 'relb', 10: 'rwri', 9: 'head',
                   12: 'rsho', 13: 'lsho', 14: 'lelb', 15: 'lwri'}

mpii_template = dict([(mpii_idx_to_jnt[i], []) for i in range(16)])

path = "D:\MPII Human Pose"
annot_file = "/Annotations/mpii_human_pose_v1_u12_1.mat"
test_image = "000001163.jpg"

ml_mpii = loadmat(path+annot_file, struct_as_record=False)['RELEASE'][0,0]
num_images = annotation_mpii = ml_mpii.__dict__['annolist'][0].shape[0]

print(num_images)




















class GrayScaleDS(Dataset):
    def __init__(self, path, transform=None, sample_size = 16) :
       # self.imgs = random.sample(os.listdir(path),sample_size)
        self.transform = transform
        self.path = path

    def __len__(self):
        return(len(self.imgs))

    def __getitem__(self, idx):
        if torch.is_tensor(idx) :
            idx = idx.tolist()

        if self.transform:
            gray = self.transform(gray)
            color = self.transform(color)
        sample = {'train':gray, 'truth':color, 'name':self.imgs[idx]}

        return sample