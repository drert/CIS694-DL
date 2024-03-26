import numpy as np

# params
r = 3
n = 100000

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

print("User Entered Results:")
print(log_2_monte(r,n))
print()
print("Results for 5,10,15 as requested:")
print(log_2_monte(5,n))
print(log_2_monte(10,n))
print(log_2_monte(15,n))