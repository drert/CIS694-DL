import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from colorization.colorizers import *
import os
from colorization.colorizers import *


# this code is an adaptation of the DCGAN implementation in PyTorch documentation
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html


# dataset which preprocesses using the CIC model. used in dataloader for training the DCGAN
class CIC_Processed_Dataset(Dataset) :
    def __init__(self, dir, img_size=256, transform=None) :
        self.img_list = os.listdir(dir)
        self.transform = transform
        self.dir = dir
        self.img_size = img_size
        self.CICmodel = eccv16(pretrained=True)

    def __len__(self) :
        return len(self.img_list)

    def __getitem__(self, idx) :
        if torch.is_tensor(idx) :
            idx = idx.tolist()
        img_name = self.img_list[idx]
        gt_img = load_img(self.dir + img_name) #, self.transform)
        (_, tens_l_rs, tens_ab_rs ) = preprocess_img(gt_img, HW=(self.img_size, self.img_size))

        model_res = self.CICmodel(tens_l_rs)
        tens_ab_rs = torch.squeeze(tens_ab_rs, 0)
        tens_l_rs  = torch.squeeze(tens_l_rs, 0)
        model_res = torch.squeeze(model_res, 0)

        return (tens_ab_rs, tens_l_rs, model_res)

# CUSTOM LOSS FUNCTIONS IN FOR PIX2PIX IMPROVEMENT
# https://github.com/akanametov/pix2pix
class GenLoss(nn.Module) :
    def __init__(self, alpha=1) :
        super(GenLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.MSELoss()

    def forward(self, fake, real, fake_pred) :
        fake_target = torch.ones_like(fake_pred)
        loss = self.bce(fake_pred, fake_target) + self.alpha * self.l1(fake, real)
        return loss
    
class DisLoss(nn.Module) :
    def __init__(self) :
        super(DisLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, fake_pred, real_pred) :
        fake_target = torch.zeros_like(fake_pred)
        real_target = torch.ones_like(real_pred)
        fake_loss = self.loss_fn(fake_pred, fake_target)
        real_loss = self.loss_fn(real_pred, real_target)
        loss = (fake_loss + real_loss) / 2
        return loss







# NEW P2P GAN IMPLEMENTATION MERGED FROM REFERENCES
# https://www.researchgate.net/figure/Diagram-illustrating-the-Pix2Pix-network-architecture_fig1_354884576
# https://neptune.ai/blog/pix2pix-key-model-architecture-decisions
# https://github.com/akanametov/pix2pix
class Encode(nn.Module) :
    def __init__(self, in_chan, out_chan, kernel_size=4, stride=2, padding=1, norm=True):
        super(Encode, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size, stride, padding)

        self.bn=None
        if norm:
            self.bn=nn.BatchNorm2d(out_chan)
        
    def forward(self,x):
        z = self.lrelu(x)
        z = self.conv(z)
        if self.bn is not None:
            z = self.bn(z)
        return z

class Decode(nn.Module) :
    def __init__(self, in_chan, out_chan, kernel_size=4, stride=2, padding=1, dropout=True) : 
        super(Decode, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.deconv = nn.ConvTranspose2d(in_chan, out_chan, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_chan)

        self.drop = None
        if dropout is not None:
            nn.Dropout(0.5, inplace=True)

    def forward(self, x) :
        z = self.relu(x)
        z = self.deconv(z)
        z = self.bn(z)
        
        if self.drop is not None :
            z = self.dropout(z)
        
        return z

class Gen(nn.Module) :
    def __init__(self, UNET=True) :
        super(Gen, self).__init__()
        self.mult = 1

        # u net selects whether old data is propagated forward as in the U-Net implementation Pix-2-Pix
        if UNET :
            self.mult = 2

        # encode data inits
        self.enc1 = nn.Conv2d(2 ,32, 4, 2, 1)
        self.enc2 = Encode(32,64)
        self.enc3 = Encode(64,128)
        self.enc4 = Encode(128,256)
        self.enc5 = Encode(256,256)
        self.enc6 = Encode(256,256)
        self.enc7 = Encode(256,256)
        self.enc8 = Encode(256,256, norm=False)

        # decode data inits
        self.dec8 = Decode(256,256,dropout=True)
        self.dec7 = Decode(self.mult*256,256,dropout=True)
        self.dec6 = Decode(self.mult*256,256,dropout=True)
        self.dec5 = Decode(self.mult*256,256)
        self.dec4 = Decode(self.mult*256,128)
        self.dec3 = Decode(self.mult*128,64)
        self.dec2 = Decode(self.mult*64,32)
        self.dec1 = nn.ConvTranspose2d(self.mult * 32, 2, 4, 2, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)
        if self.mult == 2 : # if the UNET mode is enabled, contatenate the old (unencoded) data with the decoded data
            d8 = self.dec8(e8)
            d8 = torch.cat([d8, e7], dim=1)
            d7 = self.dec7(d8)
            d7 = torch.cat([d7, e6], dim=1)
            d6 = self.dec6(d7)
            d6 = torch.cat([d6, e5], dim=1)
            d5 = self.dec5(d6)
            d5 = torch.cat([d5, e4], dim=1)
            d4 = self.dec4(d5)
            d4 = torch.cat([d4, e3], dim=1)
            d3 = self.dec3(d4)
            d3 = torch.cat([d3, e2], dim=1)
            d2 = self.dec2(d3)
            d2 = torch.cat([d2, e1], dim=1)
            d1 = self.dec1(d2)

        else :
            d8 = self.dec8(e8)
            d7 = self.dec8(d8)
            d6 = self.dec8(d7)
            d5 = self.dec8(d6)
            d4 = self.dec8(d5)
            d3 = self.dec8(d4)
            d2 = self.dec8(d3)
            d1 = self.dec8(d2)

        return d1


# discriminator network
class DisBlock(nn.Module) :
    def __init__(self, in_chan, out_chan, kernel_size=4, stride=2, padding=1, norm=True) :
        super(DisBlock, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size, stride, padding)
        self.n = None
        if norm :
            self.n = nn.InstanceNorm2d(out_chan)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self,x):
        z = self.conv(x)
        if self.n is not None :
            z = self.n(z)
        z = self.lrelu(z)

        return z
    

class Dis(nn.Module) :
    def __init__(self, cond=False) :
        super(Dis, self).__init__()
        self.mult = 1
        if cond :
            self.mult = 2
        self.b1 = DisBlock(self.mult * 2, 16, norm=False)
        self.b2 = DisBlock(16, 32)
        self.b3 = DisBlock(32, 64)
        self.b4 = DisBlock(64, 128)
        self.b5 = DisBlock(128, 256)
        self.b6 = DisBlock(256, 512)
        self.b7 = DisBlock(512, 1024)
        self.b8 = nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=1)
        self.sig = nn.Sigmoid()

    def forward(self, x, cond=None) :
        if cond is not None :
            x = torch.cat([x,cond], dim=1)
        z = self.b1(x)
        z = self.b2(z)
        z = self.b3(z)
        z = self.b4(z)
        z = self.b5(z)
        z = self.b6(z)
        z = self.b7(z)
        z = self.b8(z)
        return self.sig(z)

# weight initialization according to standard from DCGAN paper
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)








# generator network - PREVIOUS ITERATION

class OLD_DCGAN_Gen(nn.Module) :
    def __init__(self) :
        super().__init__()
    
        self.main = nn.Sequential(
            nn.ConvTranspose2d(2, 8, 5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),


            nn.ConvTranspose2d(8, 16, 5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 16, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, 2, 5, stride=1, padding=2, bias=False),

        )

    def forward(self, input) :
        return self.main(input)


class OLD_DCGAN_Dis(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(2, 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(8, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 64, 4, 4, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return(self.main(input))




if __name__ == "__main__" :
    appended = Gen()
    summary(appended, input_size=(4,2,256,256))
    appended = Dis()
    summary(appended, input_size=(4,2,256,256))
    appended = OLD_DCGAN_Dis()
    summary(appended, input_size=(4,2,256,256))

