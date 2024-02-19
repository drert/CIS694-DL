import torch
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# additional parameters
dist_exp = 2


# make fake data by normal distribution (mean, std)
n_data = torch.ones(1000, 2)
x0 = torch.normal(2*n_data, 1)      # class0 x data (tensor), shape=(100, 2)
x1 = torch.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(100, 2)
x2 = torch.normal(-5*n_data, 1)     # class2 x data (tensor), shape=(100, 2)
x3 = torch.normal(5*n_data, 1)      # class3 x data (tensor), shape=(100, 2)
x4 = torch.normal(10*n_data, 1)     # class4 x data (tensor), shape=(100, 2)
x = torch.cat((x0, x1, x2, x3, x4), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating

################# Do kmeans clustering by calling the sklearn built-in function #################
cluster_num = 5
kmeans = KMeans(n_clusters=cluster_num, random_state=0, n_init='auto').fit(x.data.numpy())
# print(kmeans.cluster_centers_)

# plt.ion()   # something about plotting
# pred_y = kmeans.labels_
# plt.cla()
# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
# plt.ioff()
# plt.show()

################# 1 Implement your own kmeans clustering function myKmeans() (40 points) #################

# Write your code here for your own kmeans clustering function
def point_dist(x1: list,x2: list) :
    # take in 2 numpy vectors, return distance
    if len(x1) != len(x2) :
        print("length mismatch")
        return None
    
    diff = x1 - x2
    exp = diff ** dist_exp
    sum = torch.sum(exp)
    dist = sum ** (1/dist_exp)
    return dist

def plot_clusters(data, labels,K) :
    for mean_idx in range(K) : 
        filtered = np.array([point.data.numpy() for point,label in zip(data, labels) if label == mean_idx])
        print(filtered)
        plt.scatter(filtered[:,0], filtered[:,1])
    plt.show()
    return

def plot_full (data) :
    fixed = np.array([point.data.numpy() for point in data])
    plt.scatter(fixed[:,0],fixed[:,1])
    plt.show()
    return

def myKmeans(x, K, max_iteration=1000):
    # write your code here based on the algorithm described in class, and you cannot call other kmeans clustering packages here

    # initialize random centers 
    #   - i just select random points from the dataset to ensure that we have some reasonable start position
    idxs = torch.randperm(len(x))[:K]
    means = x[idxs].data.numpy()
    print(means)

    # initialize trackers
    iter = 0
    converged = False
    sums = np.zeros((K, x.size(dim=1))) # 
    counts = np.zeros(K)
    labels = np.zeros(x.size(dim=0))

    # initial plot
    plot_full(x)

    # start of main iteration loop
    while (iter < 10) : # max_iteration) and (converged == False) :
        # assignment step - assign each observation to the nearest mean
        for x_idx in range(x.size(dim=0)) :
            dists = [point_dist(x[x_idx], mean_i) for mean_i in means]
            min_idx = np.argmin(dists)
            # print(x_idx, "  ", iter, "  ", dists, "  ", min_idx)
            
            sums[min_idx] += x[x_idx].data.tolist()
            labels[x_idx] = min_idx
            counts[min_idx] += 1 
        
        # update step - calculate new means of the re-assigned observations in the new clusters
        new_means = np.array([sum_/count_ for sum_,count_ in zip(sums, counts)])
        print(new_means)

        # check for convergence
        if np.array_equal(means, new_means, equal_nan=False) :
            converged = True

        # checker plot
        plot_clusters(x, labels, K)
        
        # update iteration, reset trackers, update means
        iter += 1
        means = new_means
        sums = np.zeros((K, x.size(dim=1))) # 
        counts = np.zeros(K)


    center, mean_intra_cluster_distance, mean_inter_cluster_distance = 0,0,0
    return center, mean_intra_cluster_distance, mean_inter_cluster_distance



################# 2 Optimal K for your own kmeans clustering (10 points) #################

# Write your code for a loop to call your own function myKmeans() by setting cluster_number=K from 2 to 10
# print the ratio of mean_intra_cluster_distance over mean_inter_cluster_distance for each K.
# print the optimal K with minimum ratio

center,intra,inter = myKmeans(x,5)
# for i in range(10) :
#     center,intra,inter =myKmeans(x,i)
#     print(intra/inter)