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

# params
batch_size = 16
epochs = 5
sample_sets = 2
sample_size = 64 # number of images to use in training
img_size = 128

DB_PATH = "D:\IMAGE_DB\ImageNet1k\Set 1/" # path to training images

if torch.cuda.is_available() :
    device = torch.device("cuda")
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.get_device_name(0))
else :
    device = torch.device("cpu")
    print("Using CPU")

# image transformations
transformer = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((img_size, img_size))
              ])

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
                            nn.Conv2d(1, 32, 9, padding=4), # P = (F-1)/2 returns same size of image.
                            nn.ReLU(),
                            nn.Conv2d(32, 64, 6, padding=2),
                            nn.ReLU(),
                            nn.Conv2d(64, 128, 7, padding=2, stride = 4),
                            nn.ReLU()
                        )
        
        # QUARTER POOL / QUADUP
        self.qpool = nn.AvgPool2d(2, stride=2, padding=0) # 
        self.q_ups = nn.Upsample (scale_factor=2)

        # HALF POOL / DOUBLEUP
        self.hpool = nn.AvgPool2d(4, stride=4, padding=0)
        self.h_ups = nn.Upsample (scale_factor=4)

        # 1x1 CONV
        self.c_red = nn.Conv2d(128*3, 64, 1)

        ## DECODER
        self.dec = nn.Sequential(
                            nn.Upsample(scale_factor = 4),
                            nn.Conv2d(64, 16, 5, padding = 2), # P = (F-1)/2 returns same size of image.
                            nn.ReLU(),
                            nn.Conv2d(16, 3, 3, padding=1),
                            nn.ReLU()
                        )
        
# padding = 4, dilation = 2?       
    def forward(self, x):   
        encoded = self.enc(x)

        qpooled = self.qpool(encoded)
        q_upped = self.q_ups(qpooled)
        
        hpooled = self.hpool(encoded)
        h_upped = self.h_ups(hpooled)

        catted  = torch.concat((encoded, q_upped, h_upped), 1)
        onexone = self.c_red(catted)
        decoded = self.dec(onexone)
        return decoded

        # x = torch.flatten(x, 1)     # flatten all dimensions except the batch dimension
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)

# initialize network
print("Train Start")
net = Net().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=.01, momentum=0.95 )

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

PATH = './GRAYNET.pth'
torch.save(net.state_dict(), PATH)

net = Net()
net.load_state_dict(torch.load(PATH))
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


# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         # calculate outputs by running images through the network
#         outputs = net(images)
#         # the class with the highest energy is what we choose as prediction
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print('Accuracy of the network on the 10000 test images: %d %%' % (
#         100 * correct / total))

# # prepare to count predictions for each class
# correct_pred = {classname: 0 for classname in classes}
# total_pred = {classname: 0 for classname in classes}

# # again no gradients needed
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         outputs = net(images)
#         _, predictions = torch.max(outputs, 1)
#         # collect the correct predictions for each class
#         for label, prediction in zip(labels, predictions):
#             if label == prediction:
#                 correct_pred[classes[label]] += 1
#             total_pred[classes[label]] += 1

# # print accuracy for each class
# for classname, correct_count in correct_pred.items():
#     accuracy = 100 * float(correct_count) / total_pred[classname]
#     print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
#                                                          accuracy))
