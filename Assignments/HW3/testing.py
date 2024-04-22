import torch
import numpy as np
import matplotlib.pyplot as plt
from training import Net, transformer, criterion
import os
import cv2

# This script assumes that the path entered below contains the testing images.
# assumed that the form of the ground truth images is "name-gt.xxx" 
             # with associated grayscale testing file as "name.xxx" 

TESTING_PATH = "./testing_data/"

MODEL_PATH = "./GRAYNET.pth"
testing_images = []
grtruth_images = []
for img in os.listdir(TESTING_PATH) :
    if img[-7:-3] == "-gt." :
        gr = cv2.imread(TESTING_PATH + img)
        gr = cv2.cvtColor(gr,cv2.COLOR_BGR2RGB)
        gr = transformer(gr)
        grtruth_images.append(gr)
        
    else :
        te = cv2.imread(TESTING_PATH + img)
        te = cv2.cvtColor(te,cv2.COLOR_BGR2GRAY)
        te = transformer(te)
        testing_images.append(te)
batch_size = len(testing_images)

net = Net()
net.load_state_dict(torch.load(MODEL_PATH))

correct = 0
total = 0
with torch.no_grad():
    gts = torch.stack(grtruth_images)   
    tes = torch.stack(testing_images)   

    outputs = net(tes)
    loss = criterion(outputs, gts)
    print("Loss:", loss)
        # drawing last trained image batch
        
    fig,ax = plt.subplots(batch_size,3)
    ax[0,0].title.set_text("input image")
    ax[0,1].title.set_text("resulting image")
    ax[0,2].title.set_text("ground truth image")
    for i in range(batch_size):
        in_img   =    tes[i]
        res_img  =   outputs[i]
        gt_img   =    gts[i]
        ax[i,0].imshow(in_img.detach().permute(1,2,0).cpu(), cmap='gray')
        ax[i,1].imshow(res_img.detach().permute(1,2,0).cpu())
        ax[i,2].imshow(gt_img.detach().permute(1,2,0).cpu())
    plt.show()