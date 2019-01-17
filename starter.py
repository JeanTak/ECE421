import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def loadData():
	with np.load('notMNIST.npz') as data:
		Data, Target = data['images'], data['labels']
		posClass = 2
		negClass = 9
		dataIndx = (Target==posClass) + (Target==negClass)
		Data = Data[dataIndx]/255.
		Target = Target[dataIndx].reshape(-1, 1)
		Target[Target==posClass] = 1
		Target[Target==negClass] = 0
		np.random.seed(421)
		randIndx = np.arange(len(Data))
		np.random.shuffle(randIndx)
		Data, Target = Data[randIndx], Target[randIndx]
		trainData, trainTarget = Data[:3500], Target[:3500]
		validData, validTarget = Data[3500:3600], Target[3500:3600]
		testData, testTarget = Data[3600:], Target[3600:]
	return trainData, validData, testData, trainTarget, validTarget, testTarget



def MSE(W, b, x, y, reg):
	# ref: https://chunml.github.io/ChunML.github.io/tutorial/Regularization/
	# ref: https://en.wikipedia.org/wiki/Linear_regression

	# the predicted y
	predicted = [sum(W * x[i]) + b[i] for i in range(len(x))]

	# the amount of data
	m = len(predicted)
	
	# calculate regularized MSE
	mse = sum((predicted - y) ** 2) / (2 * m) + sum(W ** 2) * reg / (2 * m)
	return mse

# a = [[1,2,3,4,5],[1,2,3,4,10],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]	
# MSE(np.array([1,2,3,4,5]), np.array([1,2,3,4,5]), np.array(a), np.array([1,2,3,4,5]), 1)



def gradMSE(W, b, x, y, reg):
	# ref: https://chunml.github.io/ChunML.github.io/tutorial/Regularization/

	# the predicted y
	predicted = [sum(W * x[i]) + b[i] for i in range(len(x))]

	# the amount of data
	m = len(predicted)

	# GRADIENT OF WEIGHT
	grad_weight = np.array([])
	for j in range(len(W)):

		temp = reg * W[j]

		for i in range(m): temp += (predicted[i] - y[i]) * x[i][j]
		
		temp = temp / m

		np.append(grad_weight, temp)

	# GRADIENT OF BIAS
	grad_bias = sum([predicted - y]) / m

	return grad_weight, grad_bias
	
data = np.load("data.npy", mmap_mode='r')
target = np.load("target.npy", mmap_mode='r')
print(data[1][1])
print(target[1][1])


# def crossEntropyLoss(W, b, x, y, reg):
#     # Your implementation here

# def gradCE(W, b, x, y, reg):
#     # Your implementation here

# def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
	

# def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
#     # Your implementation here


