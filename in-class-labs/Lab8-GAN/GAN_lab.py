"""
GAN Lab: learning to draw Sin(x) by GAN
06/03/2021 tested on PyTorch 1.8.1
This code is modified from https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/406_GAN.py

Dependencies:
torch: 0.4
numpy
matplotlib
"""
# Question1: Please search for To-do1 and To-do2 and finish them
# Question2: If it works, congratulations! Then, please think about the logic for loss functions
# Question3: What will be prob_artist0 when D is converged?

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible
# np.random.seed(1)

# Hyper Parameters
BATCH_SIZE = 64
LR_G = 0.00001  # learning rate for generator
LR_D = 0.00001  # learning rate for discriminator
N_IDEAS = 5  # think of this as number of ideas for generating an art work (Generator)
ART_COMPONENTS = 50  # it could be total point G can draw in the canvas
PAINT_POINTS = np.vstack([np.linspace(-10, 10, ART_COMPONENTS) for _ in range(BATCH_SIZE)])


def artist_works():  # painting real Sin(x) from a real artist
    paintings = np.sin(PAINT_POINTS)
    paintings = torch.from_numpy(paintings).float()
    return paintings


G = nn.Sequential(  # Generator
    nn.Linear(N_IDEAS, 128),  # random ideas (could from normal distribution)
    nn.ReLU(),
    nn.Linear(128, ART_COMPONENTS),  # making a painting from these random ideas
)

D = nn.Sequential(  # Discriminator
    nn.Linear(ART_COMPONENTS, 128),  # receive art work either from the famous artist or a newbie like G
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid(),  # tell the probability that the art work is made by artist
)

opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)
criterion = nn.MSELoss()

plt.ion()  # something about continuous plotting

for step in range(10001):
    artist_paintings = artist_works()  # real painting from artist
    G_ideas = torch.randn(BATCH_SIZE, N_IDEAS, requires_grad=True)  # random ideas
    G_paintings = G(G_ideas)  # fake painting from G (random ideas)
    prob_artist1 = D(G_paintings)  # D try to reduce this prob (Fake by G)
    # To-do1: please write your generator loss here: G_loss = ...
    G_loss = torch.log(criterion(G_paintings, artist_paintings))

    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()

    prob_artist0 = D(artist_paintings)  # G try to increase this prob (Real)
    prob_artist1 = D(G_paintings.detach())  # D try to reduce this prob (Fake by G)
    # To-do2: please write your discriminator loss here: D_loss = ...
    D_loss = torch.log(1-criterion(prob_artist0, prob_artist1))
    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)  # reusing computational graph
    opt_D.step()

    if step % 50 == 0:  # plotting
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='Generated Sin(x)', )
        plt.plot(PAINT_POINTS[0], np.sin(PAINT_POINTS[0]), c='#FF9359', lw=3, label='Real Sin(x)')
        plt.text(-10, 4.5, 'D prob=%.2f ' % prob_artist0.data.numpy().mean(),
                 fontdict={'size': 13})
        plt.text(-10, 4, 'D loss= %.2f ' % -D_loss.data.numpy(), fontdict={'size': 13})
        plt.text(-10, 3.5, 'step = %d ' % step, fontdict={'size': 13})
        plt.ylim((-5, 5))
        plt.legend(loc='upper right', fontsize=13)
        plt.draw()
        plt.pause(0.01)

plt.ioff()
plt.show()