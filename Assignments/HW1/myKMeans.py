import torch
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# additional parameters
draw_results = False
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
def point_dist(x1, x2, numpy_arr= False) :
    # take in 2 tensors, return distance
    if len(x1) != len(x2) :
        print("length mismatch")
        return None
    
    diff = x1 - x2
    exp = diff ** dist_exp

    if numpy_arr : # conditional for when calculating means later (using numpy arrays)
        sum = np.sum(exp)
    else :
        sum = torch.sum(exp)

    dist = sum ** (1/dist_exp)
    return dist

def plot_clusters(data, labels,K) :
    for mean_idx in range(K) : 
        filtered = np.array([point.data.numpy() for point,label in zip(data, labels) if label == mean_idx])
        plt.scatter(filtered[:,0], filtered[:,1])
    plt.show()
    return

def plot_full (data) :
    fixed = np.array([point.data.numpy() for point in data])
    plt.scatter(fixed[:,0],fixed[:,1])
    plt.show()
    return

def inter_intra_calc(data, labels, K, means) :
    # intracluster calculation
    inter,intra = 0,0

    tot_intra = 0
    counted = 0
    for mean_idx in range(K) : 
        filtered = np.array([point.data.numpy() for point,label in zip(data, labels) if label == mean_idx])
        fil_len = len(filtered)
        for i in range(fil_len) :
            for j in range(fil_len-i-1) :
                dist = point_dist(filtered[i],filtered[j+i+1],numpy_arr=True)
                tot_intra += dist
                counted += 1
    intra = tot_intra / counted
    # print("COUNTED: ", counted)

    # intercluster calculation
    tot_inter = 0
    counted = 0
    considered_vals = []
    for i in range(K) :
        for j in range(K-i-1) :
            # print(i,j+i+1)
            dist = point_dist(means[i],means[j+i+1], numpy_arr=True)
            tot_inter += dist
            counted += 1
            # if dist in considered_vals :
            #     print("DOUBLE COUNT")
            # else :
            #     considered_vals.append(dist)
    inter = tot_inter / counted
    # print("COUNTED: ", counted)
    return inter, intra

def myKmeans(x, K, max_iteration=1000):
    # write your code here based on the algorithm described in class, and you cannot call other kmeans clustering packages here

    # initialize random centers 
    #   - i sample points (without replacement) from the dataset to ensure that we have some reasonable start position
    idxs = torch.randperm(len(x))[:K]
    means = x[idxs].data.numpy()

    # initialize trackers
    iter = 0
    converged = False
    sums = np.zeros((K, x.size(dim=1))) # 
    counts = np.zeros(K)
    labels = np.zeros(x.size(dim=0))

    # initial plot
    # plot_full(x)

    # start of main iteration loop
    while (iter < max_iteration) and (converged == False) :
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
        
        # check for convergence
        if np.array_equal(means, new_means, equal_nan=False) :
            converged = True
        
        # tester printouts, plotting
        # if iter % 5 == 0:
        #     print(iter)
        # if iter % 10 == 0 :
        #     plot_clusters(x, labels, K)
            

        # update iteration, reset trackers, update means
        iter += 1
        means = new_means
        sums = np.zeros((K, x.size(dim=1))) # 
        counts = np.zeros(K)

    if (draw_results) :
        plot_clusters(x, labels, K)
    
    center = means
    mean_inter_cluster_distance, mean_intra_cluster_distance = inter_intra_calc(x, labels, K, means)
    return center, mean_intra_cluster_distance, mean_inter_cluster_distance



################# 2 Optimal K for your own kmeans clustering (10 points) #################

# Write your code for a loop to call your own function myKmeans() by setting cluster_number=K from 2 to 10
# print the ratio of mean_intra_cluster_distance over mean_inter_cluster_distance for each K.
# print the optimal K with minimum ratio

best_ratio = 1000000
best_K = 0
for i in range(2,10) :
    center,intra,inter =myKmeans(x,i)
    if intra/inter < best_ratio :
        best_K = i
        best_ratio = intra/inter
    print("When K =", i, "mean intra-cluster distance:", intra, 
                         ",mean inter-cluster distance:", inter, 
                         ", ratio:", intra/inter)

print("Best intra/inter-cluster ratio achieved at K =", best_K, "ratio =", best_ratio)