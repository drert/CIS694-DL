import numpy as np

# params
r = 3
n = 10000

def log_2_monte(R,N) :
    green_count = 0
    for _ in range(N) :
        x = np.random.rand() * R
        y = np.random.rand() * R
        if y <= np.log2(x) :
            green_count += 1
    # print(green_count)
    A = (green_count / N) * R * R
    # print((green_count / N) * R * R)
    return A


print(log_2_monte(r,n))
sum = 0
for i in range(1000) :
    sum += log_2_monte(r,n)
    if i%10 == 0 :
        print(i)
print(sum/100)