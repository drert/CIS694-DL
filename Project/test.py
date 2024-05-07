import torch
import numpy as np
import matplotlib.pyplot as plt
from GAN import *
from colorization.colorizers import *


if __name__ == "__main__" :
    MODEL_PATH = './G_PREV.pth'
    TESTING_PATH = "testing_images/"
    G = Gen()
    G.load_state_dict(torch.load(MODEL_PATH))
    CIC = eccv16(pretrained=True)
    testing_images = []
    grtruth_images = []

    for img in os.listdir(TESTING_PATH) :
        gr = load_img(TESTING_PATH + img)
        (tens_l_orig, tens_l_rs, tens_ab_rs) = preprocess_img(gr)
        testing_images.append(tens_l_rs)
        grtruth_images.append(tens_ab_rs)
    
    batch_size = len(testing_images)
    with torch.no_grad() :
        print("test")
        gts = torch.concat(grtruth_images)
        tes = torch.concat(testing_images)

        res = CIC(tes)
        fin = G(res)

    
        # drawing last trained image batch
        fig,ax = plt.subplots(batch_size,4)

        ax[0,0].title.set_text("input")
        ax[0,1].title.set_text("ECCV16")
        ax[0,2].title.set_text("+ GAN")
        ax[0,3].title.set_text("ground truth")
        for i in range(batch_size):
            in_img   =    tes[i].detach().permute(1,2,0).cpu()
            res1_img  =   postprocess_tens(torch.stack((tes[i],)),torch.stack((res[i],)))
            fin_img =     postprocess_tens(torch.stack((tes[i],)),torch.stack((fin[i],)))
            gt_img   =    postprocess_tens(torch.stack((tes[i],)),torch.stack((gts[i],)))
            ax[i,0].imshow(in_img, cmap='gray')
            ax[i,0].axis("off")
            ax[i,1].imshow(res1_img)
            ax[i,1].axis("off")
            ax[i,2].imshow(fin_img)
            ax[i,2].axis("off")
            ax[i,3].imshow(gt_img)
            ax[i,3].axis("off")
        plt.subplots_adjust(wspace=0,hspace=0)
        plt.show()

