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
    print(device)
    epochs = 2
    image_size = 256
    train_dir = "training_images/"

    # initialize generator
    G = Gen().to(device)
    # G.apply(weights_init)

    # initalize discriminator
    D = Dis().to(device)
    # D.apply(weights_init)

    # binary cross-entropy loss for image comparison
    criterion = nn.BCELoss()

    # generate 2 optimizers for G, D
    optimG = optim.Adam(G.parameters(), lr=lr, betas = (beta1, 0.999))
    optimD = optim.Adam(D.parameters(), lr=lr, betas = (beta1, 0.999))

    # training information storage
    img_list = []
    G_losses = []
    D_losses = []
    G_times = []
    D_times = []
    tween_times = []
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
        tween_start = time.time()
        for i, data in enumerate(dataloader,0) :
            # unpack loaded data
            gt, orig_input, inputs = data
            inputs = inputs.to(device)
            gt = gt.to(device)
            # print(gt.shape, orig_input.shape, inputs.shape)

            # timer step
            tween_end = time.time()
            tween_times.append(tween_end - tween_start)
            D_start = time.time()

            # # D network update
            D.zero_grad()

            # train on all-real batch
            # generate real labels, compare to results of discrim.
            size = gt.size(0)
            label = torch.full((size,), real_label, dtype=torch.float, device=device)
            output = D(gt).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward() # propagate gradient
            D_x = output.mean().item()

            # train on all-fake batch
            # generate fake labels, compare to results of discrim.
            fakes = G(inputs)
            label.fill_(fake_label)
            output = D(fakes.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward() # propagate gradient
            D_G_z1 = output.mean().item() 

            errD = errD_fake + errD_real
            optimD.step()

            # timer step
            D_end = time.time()
            G_start = time.time()

            # G network update
            G.zero_grad()
            label.fill_(real_label)

            # recalculate steps given D was updated
            output = D(fakes).view(-1)
            errG = criterion(output, label)
            errG.backward() # propagate gradient
            D_G_z2 = output.mean().item()
            optimG.step()

            # timer step
            G_end = time.time()
            tween_start = time.time()

            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # value checker printout
            print("GT Mean: " + str(torch.mean(gt).item())
              + "; GT Max: " + str(torch.max(gt).item())
              + "; GT Min: " + str(torch.min(gt).item())
              + "\nGen Mean: " + str(torch.mean(fakes).item())
              + "; Gen Max: " + str(torch.max(fakes).item())
              + "; Gen Min: " + str(torch.min(fakes).item())
            )

            if i % 3 == 0 :
                print(epoch, i)

            G_times.append(G_end - G_start)
            D_times.append(D_end - D_start)

        print("G exec. time: " + str(np.average(G_times)) + "; D exec. time: " + str(np.average(D_times)) + "; Tween exec. time: " + str(np.average(tween_times)))

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

