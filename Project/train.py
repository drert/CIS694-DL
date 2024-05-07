from GAN import *
from colorization.colorizers import *
import torch, torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
from torchvision import transforms
import time


if __name__ == "__main__" :
    workers = 0 ## NON FUNCTIONAL
    lr = .0001
    batch_size = 2
    beta1 = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    epochs = 1
    alpha = .01
    image_size = 256
    train_dir = "training_images/"

    # initialize generator
    G = Gen().to(device)
    # G.apply(weights_init)

    # initalize discriminator
    D = Dis().to(device)
    # D.apply(weights_init)

    # binary cross-entropy loss for image comparison
    g_crit = GenLoss(alpha=alpha)
    d_crit = DisLoss()

    # generate 2 optimizers for G, D
    optimG = optim.Adam(G.parameters(), lr=lr, betas = (beta1, 0.999))
    optimD = optim.Adam(D.parameters(), lr=lr, betas = (beta1, 0.999))

    # training information storage
    img_list = []
    G_losses = []
    D_losses = []
    # G_times = []
    # D_times = []
    # tween_times = []
    iters = 0

    # launch dataset
    transformer = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((image_size, image_size))
                ])
    dataset = CIC_Processed_Dataset(train_dir, image_size, transform = transformer)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers = workers)

    print("Start of training loop") 

    real_label = 0
    fake_label = 1

    for epoch in range(epochs) :
        # tween_start = time.time()
        for x, data in enumerate(dataloader,0) :
            # unpack loaded data
            gt, orig_input, inputs = data
            inputs = inputs.to(device).detach()
            gt = gt.to(device)
            # print(gt.shape, orig_input.shape, inputs.shape)

            # G GEN
            fakes = G(inputs)
            pred_fakes = D(fakes)
            g_loss = g_crit(fakes, gt, pred_fakes)
        
            # G STEP
            optimG.zero_grad()
            g_loss.backward()
            optimG.step()
            

            # D GEN
            fakes = G(inputs).detach()
            pred_real =  D(gt)
            pred_fakes = D(fakes)
            d_loss = d_crit(pred_fakes, pred_real)


            # D STEP
            optimD.zero_grad()
            d_loss.backward()
            optimD.step()

 




            # tally losses
            print(g_loss.item())
            print(d_loss.item())
            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())



            # incremental print
            if x % 3 == 0 :
                print(epoch, x)

                # value checker printout
                print("GT Mean: " + str(torch.mean(gt).item())
                  + "; GT Max: " + str(torch.max(gt).item())
                  + "; GT Min: " + str(torch.min(gt).item())
                  + "\nGen Mean: " + str(torch.mean(fakes).item())
                  + "; Gen Max: " + str(torch.max(fakes).item())
                  + "; Gen Min: " + str(torch.min(fakes).item())
                )
    # save model
    # save results
    PATH = './G_PREV.pth'
    torch.save(G.state_dict(), PATH)

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training lr =" + str(lr))
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

