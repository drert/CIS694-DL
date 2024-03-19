# Linear regression with single variable: y= theta_0 + theta_1*x

import numpy as np
import matplotlib.pyplot as plt

####### Set hyperparameters #######
epoch = 100
learning_rate = 0.00001

####### Generate training data: assume that you got it from somewhere #######
m = 300
gt_theta_0 = 9.5
gt_theta_1 = 3.5
#intX = print(np.random.randint(low=1, high=100, size=m))
x = np.random.uniform(2.5, 100.5, size=(m,))
noise = np.random.uniform(-10, 10, size=(m,))
y = gt_theta_0 + gt_theta_1*x + noise

####### Implement the optimization algorithm (introduced in class) to get predicted theta_0 and theta_1 #######
theta_0 = 10
theta_1 = 10
loss = []

	##############Please write your code here#######
# h(theta) - predition 
def calc_loss(x,t0,t1,y) :
    return np.sum(((t0 + t1 * x ) - y)**2) / (2 * m)

for e in range(epoch) :
	delta_theta_0 = np.sum(((theta_0 + theta_1 * x ) - y) / m)
	delta_theta_1 = np.sum(x * ((theta_0 + theta_1 * x ) - y) / m)
    
	theta_0 -= delta_theta_0 * learning_rate
	theta_1 -= delta_theta_1 * learning_rate

	loss.append(calc_loss(x,theta_0,theta_1,y))
	##############Please write your code here#######


####### Show loss decreasing and print the ground-truth and predicted theta_0 and theta_1 #######
print('Ground-truth theta_0:' + str(gt_theta_0) + '\n')
print('Ground-truth theta_1:' + str(gt_theta_1) + '\n')
print('Predicted theta_0:' + str(theta_0) + '\n')
print('Predicted theta_1:' + str(theta_1) + '\n')
plt.plot(np.array(range(epoch)), loss, marker='x', color='b')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Linear Regression with Single Variable')
plt.show()







