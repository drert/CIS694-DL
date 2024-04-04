# CIS694 - Deep Learning - Spring 2024
# Alexander Sukennyk - 2717393
# Assignment 2 submission main file

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.io as io
import os
from tifffile import imread

# FOR INSTRUCTOR VALIDATION:
#   IT IS EXPECTED THAT THE RELATIVE PATH "database/" IS WHERE THE IMAGES WILL BE LOCATED
#   THE TESTING IMAGES ARE MANUALLY ENTERED IN THE ARRAY BELOW FOR EASE OF CHANGE


# load resnet18 from pytorch, recreate exclusing final step (Fully connected layer)
rn18 = models.resnet18(weights='IMAGENET1K_V1')
extractor = nn.Sequential(*list(rn18.children())[:-1])

# data store init
loaded_images = []
train_names = []

# load images from database folder, correct to 
img_folder = "database/"
for img in os.listdir(img_folder) :
    if img[-4:] == ".png" :
        loaded_images.append(io.read_image(img_folder + img))
    else :
        loaded_images.append(torch.from_numpy(imread(img_folder + img)))
    train_names.append(img)

# generate transformer to preprocess images to correct size
init_tran = transforms.Compose([transforms.ToPILImage("RGB"),
                                transforms.Resize((224,224)),
                                transforms.ToTensor()])

# enumerating loop for preprocessing
extracted_features = []
for idx,li in enumerate(loaded_images) :

    # rotate dimensions to match
    if li.size()[2] == 3 :
        loaded_images[idx] = li.permute(2,0,1)

    # transform images
    loaded_images[idx] = init_tran(loaded_images[idx])

# stack resulting tensors together, 
transformed_images = torch.stack(list(loaded_images))
extracted_features = rn18(transformed_images)

# import, generate K = 2 KMeans from sci-kit learn
from sklearn.cluster import KMeans 
import numpy as np
km = KMeans(n_clusters=2,n_init="auto")
ret = km.fit_predict(extracted_features.detach().numpy())
print(np.shape(km.cluster_centers_))
for i,pred in enumerate(ret) :
    print(train_names[i],":",pred)


# preprocess 2 training cases using previously assigned names