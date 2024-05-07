import os
import shutil

img_name_file = "D:/CUB-200-2011/CUB_200_2011/images.txt"
split_file = "D:/CUB-200-2011/CUB_200_2011/train_test_split.txt"
image_dir = "D:/CUB-200-2011/CUB_200_2011/images/054.Blue_Grosbeak/"
train_dir = "training_images/"
test_dir = "testing_images/"

tt_labels = {}
direct_labels = {}
with open(split_file, 'r') as f :
    for line in f.readlines() :
        split = line.split()
        tt_labels[split[0]] = split[1]

with open(img_name_file, 'r') as f:
    for line in f.readlines() :
        split = line.split()
        filename = split[1].split("/")[1]
        print(filename)
        direct_labels[filename] = tt_labels[split[0]]

count = 0
for img_name in os.listdir(image_dir) :
    tt = direct_labels[img_name]
    if tt == '0' and count < 5 :
        dest = test_dir
        count += 1
    else :
        dest = train_dir
    shutil.copy(image_dir+img_name, dest)