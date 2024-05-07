from colorization.colorizers import *
import colorization.colorizers.util as ut
from torchinfo import summary
import cv2
from torchvision import transforms
import os, torch
import matplotlib.pyplot as plt
import numpy as np



img_size = 256
TESTING_PATH = "imgs_for_fig/"
transformer = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((img_size, img_size))
            ])
model1 = eccv16(pretrained=True)
#summary(model)
model2 = siggraph17(pretrained=True)
#summary(model)


testing_images = []
grtruth_images = []

for img in os.listdir(TESTING_PATH) :
    gr = load_img(TESTING_PATH + img)
    (tens_l_orig, tens_l_rs) = preprocess_img(gr, HW=(256,256))
    
    testing_images.append(tens_l_rs)
    grtruth_images.append(transformer(gr))

batch_size = len(testing_images)

with torch.no_grad():
    gts = torch.stack(grtruth_images)   
    tes = torch.concat(testing_images)   
    
    res1 = model1(tes)
    res2 = model2(tes)

        # drawing last trained image batch
    fig,ax = plt.subplots(batch_size,4)

    ax[0,0].title.set_text("input")
    ax[0,1].title.set_text("ECCV16")
    ax[0,2].title.set_text("SIGGRAPH17")
    ax[0,3].title.set_text("ground truth")
    for i in range(batch_size):
        in_img   =    tes[i].detach().permute(1,2,0).cpu()
        res1_img  =   ut.postprocess_tens(torch.stack((tes[i],)),torch.stack((res1[i],)))
        res2_img  =   ut.postprocess_tens(torch.stack((tes[i],)),torch.stack((res2[i],)))
        gt_img   =    gts[i].detach().permute(1,2,0).cpu()
        ax[i,0].imshow(in_img, cmap='gray')
        ax[i,0].axis("off")
        ax[i,1].imshow(res1_img)
        ax[i,1].axis("off")
        ax[i,2].imshow(res2_img)
        ax[i,2].axis("off")
        ax[i,3].imshow(gt_img)
        ax[i,3].axis("off")
    plt.subplots_adjust(wspace=0,hspace=0)
    plt.show()
