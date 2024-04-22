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

# params for instructor check
reintroduce = False
DB_PATH = "D:\IMAGE_DB\ImageNet1k\Set 1/" # path to training images
img_size = 128 # as specified in architecture -- can be changed

# params - please lower or raise if re-testing for speed
batch_size = 64
sample_sets = 4 # number of times to sample database for {sample size}# images
sample_size = 1280
epochs = 4 # number of train loops = epochs * sample_sets
draw_final_test = True # True only if you want to see testing example
lr = 0.005
momentum = 0.9

transformer = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((img_size, img_size))
            ])
criterion = nn.MSELoss()

# my attempt at creating a custom dataset --> dataloader
class GrayScaleDS(Dataset):
    def __init__(self, path, transform=None, sample_size = 16) :
        self.imgs = random.sample(os.listdir(path),sample_size)
        self.transform = transform
        self.path = path

    def __len__(self):
        return(len(self.imgs))

    def __getitem__(self, idx):
        if torch.is_tensor(idx) :
            idx = idx.tolist()

        img = self.imgs[idx]
        color = cv2.cvtColor(cv2.imread(self.path + img), cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)

        if self.transform:
            gray = self.transform(gray)
            color = self.transform(color)
        sample = {'train':gray, 'truth':color, 'name':self.imgs[idx]}

        return sample


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()     
        ## ENCODER
        self.enc = nn.Sequential(
                            nn.Conv2d(1, 16, 3, padding=1), # P = (F-1)/2 returns same size of image.
                            nn.LeakyReLU(),
                        #    nn.LayerNorm((128,128)),
                            nn.Conv2d(16, 32, 4, padding=1),
                            nn.LeakyReLU(),
                            nn.LayerNorm((127,127)),
                            nn.Conv2d(32, 64, 5, padding=1, stride=4),
                            nn.LeakyReLU(),
                        #    nn.LayerNorm((32,32))
                            nn.Tanh()
                        )
        
        # QUARTER POOL / QUADUP
        self.qpool = nn.MaxPool2d(2, stride=2, padding=0) # 
        self.q_ups = nn.Upsample (scale_factor=2)

        # HALF POOL / DOUBLEUP
        self.hpool = nn.AvgPool2d(4, stride=4, padding=0)
        self.h_ups = nn.Upsample (scale_factor=4)

        # 1x1 CONV
        self.c_red = nn.Conv2d(64*3, 64, 1)

        ## DECODER
        self.dec = nn.Sequential(
                            nn.Upsample(scale_factor = 8),
                            nn.AvgPool2d(2, stride=2, padding=0),
                            nn.Conv2d(64, 16, 5, padding = 2), # P = (F-1)/2 returns same size of image.
                            nn.LeakyReLU(),
                        #    nn.LayerNorm((128,128)),
                            nn.Tanh(),
                            nn.Conv2d(16, 3, 3, padding=1),
                            nn.LeakyReLU())
        ## JOINER
        self.joiner = nn.Conv2d(4, 3, 1, padding=0)

# padding = 4, dilation = 2?       
    def forward(self, x):   
        # ENCODER
        e = self.enc(x)

        # QUARTER SIZE BRANCH
        q = self.qpool(e)
        q = self.q_ups(q)

        # HALF SIZE BRANCH
        h = self.hpool(e)
        h = self.h_ups(h)

        #CONCAT, DECODE
        c = torch.concat((e, q, h), 1)
        oxo = self.c_red(c)
        d = self.dec(oxo)

        c = torch.concat((d,x),1)
        j = self.joiner(c)

        if reintroduce :
            return j
        else :
            return d

# start of main function execution
if __name__ == "__main__" :

    if torch.cuda.is_available() :
        device = torch.device("cuda")
        print("Using Device:" + torch.cuda.get_device_name(0))
    else :
        device = torch.device("cpu")
        print("Using CPU")

    # image transformations
    transformer = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((img_size, img_size))
                ])

    # initialize network
    print("Train Start")
    net = Net().to(device)

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum )

    total_batches = int(sample_size / batch_size)
    if total_batches < 10:
        total_batches = 10
    running_loss = 0.0
    for s in range(sample_sets) :
        dataset = GrayScaleDS(DB_PATH, transformer, sample_size)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        for e in range(epochs) :
            for i, data in enumerate(loader):

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels, img_names = data["train"].to(device), data["truth"].to(device), data["name"]

                # forward
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % (int(total_batches/10)) == 0:  #
                    print('[%d, %d, %5d] loss: %.10f' % (s+1, e+1, i+1, running_loss/(sample_size/10)))
                    running_loss = 0.0

    # save results
    PATH = './GRAYNET.pth'
    torch.save(net.state_dict(), PATH)

    # drawing last trained image batch
    for i in range(batch_size):
        in_img   =    inputs[i]
        res_img  =   outputs[i]
        gt_img   =    labels[i]
        name_img = img_names[i]
        print(name_img)
        fig,ax = plt.subplots(1,3)
        fig.suptitle(name_img)
        ax[0].imshow(in_img.detach().permute(1,2,0).cpu(), cmap='gray')
        ax[0].title.set_text("input image" + str(in_img.shape))
        ax[1].imshow(res_img.detach().permute(1,2,0).cpu())
        ax[1].title.set_text("resulting image" + str(res_img.shape))
        ax[2].imshow(gt_img.detach().permute(1,2,0).cpu())
        ax[2].title.set_text("ground truth image" + str(gt_img.shape))
        plt.show()
        if i == 3:
            break
