# CIS694 - Deep Learning - Spring 2024
# Alexander Sukennyk - 2717393
# Assignment 2 submission main file

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.io as io
import os
import numpy as np
from tifffile import imread
import random

# FOR INSTRUCTOR VALIDATION:
#   IT IS EXPECTED THAT THE RELATIVE PATH "database/" IS WHERE THE TRAINING IMAGES WILL BE LOCATED
#   THE TESTING IMAGES ARE MANUALLY ENTERED IN THE ARRAY BELOW - I did not want to assume where the testing images would be.
test_files = ["testing1.png", "testing2.png"]
K_clusters = 2

# distance metric and min dist calculation for step 6
def dist(x : np.ndarray, y : np.ndarray) :
    return np.linalg.norm(y-x)

def min_dist_idx(cents, test) :
    dists = [dist(cent, test) for cent in cents]
    return np.argmin(dists)

# (1) load resnet18 from pytorch, recreate exclusing final step (Fully connected layer)
rn18 = models.resnet18(weights='IMAGENET1K_V1') # "X"
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
for idx,li in enumerate(loaded_images) :

    # rotate dimensions to match
    if li.size()[2] == 3 :
        loaded_images[idx] = li.permute(2,0,1)

    # transform images
    loaded_images[idx] = init_tran(loaded_images[idx])

# (2) stack resulting tensors together, and 
transformed_images = torch.stack(list(loaded_images))
extracted_features = rn18(transformed_images)

# (3) import, generate KMeans from sci-kit learn
from sklearn.cluster import KMeans 
import numpy as np
km = KMeans(n_clusters=K_clusters,n_init="auto")
train_res = km.fit_predict(extracted_features.detach().numpy())

# (4) obtain centroids from trained model
centroids = km.cluster_centers_ # "C_i"
# print(np.shape(km.cluster_centers_))
# for i,pred in enumerate(train_res) :
#     print(train_names[i],":",pred)


# (5) preprocess 2 testing cases using previously assigned names
loaded_images = []
for img in test_files :
    if img[-4:] == ".png" :
        loaded_images.append(io.read_image(img))
    else :
        loaded_images.append(torch.from_numpy(imread(img)))
    train_names.append(img)
for idx,li in enumerate(loaded_images) :
    if li.size()[2] == 3 :
        loaded_images[idx] = li.permute(2,0,1)
    loaded_images[idx] = init_tran(loaded_images[idx])
transformed_images = torch.stack(list(loaded_images))
extracted_features = rn18(transformed_images)

# (6,7) prediction can be done simply with sklearn's KMeans.predict() function.
#       Here, I have computed them manually as well in case this was desired (test_res_1).
test_features = extracted_features.detach().numpy()
test_res_0 = km.predict(test_features)
test_res_1 = [min_dist_idx(centroids,x) for x in test_features]
# print(test_res_0==test_res_1) # returned [True True] in testing

# Bin image names by labels for later random picks
img_bins = [[] for i in range(K_clusters)]
for idx,label in enumerate(train_res):
    img_bins[label].append(train_names[idx])
print(img_bins)

# (8) Use matplotlib to display images in requested format.
import matplotlib.pyplot as plt
from PIL import Image

fig,ax = plt.subplots(2,3)
for idx,test_img in enumerate(test_files) :
    with open(test_img, 'rb') as f:
        image = Image.open(f)
        ax[idx,0].imshow(image)
        ax[idx,0].axis('off')
        ax[idx,0].title.set_text(test_img)
    random_imgs = random.sample(set(img_bins[test_res_0[idx]]),2)
    for jdx in range(2) :
        with open(img_folder + random_imgs[jdx], 'rb') as f:
            image = Image.open(f)
            ax[idx, jdx+1].imshow(image)
            ax[idx, jdx+1].axis('off')
            ax[idx, jdx+1].title.set_text(random_imgs[jdx])

plt.show()
print("End of execution")