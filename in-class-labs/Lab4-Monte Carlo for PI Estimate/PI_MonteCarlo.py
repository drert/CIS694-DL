# https://en.wikipedia.org/wiki/Monte_Carlo_method
# Monte Carlo method to estimate the value of PI, which is 3.1415926...

import numpy as np

n = 1000000
count = 0  # m

for i in range(n):
    x = 2*np.random.rand() - 1  # generate a random number in [-1,1] where np.random.rand() is to generate a random float from 0 to 1 (uniform distribution)
    y = 2*np.random.rand() - 1
    if x**2 + y**2 <= 1:
        count = count+1


PI = 4*count/n
print(PI)