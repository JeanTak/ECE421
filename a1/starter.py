import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

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

	predicted = np.sign([sum(W * x[i] + b) for i in range(len(x))])	 # the predicted y

	m = len(predicted)	 # the amount of data

	for e in range(m):	# change -1 to 0
		if predicted[e] == -1: predicted[e] = 0
	
	# calculate regularized MSE
	mse = sum((predicted - y) ** 2) / (2 * m) + sum(W ** 2) * reg / (2 * m)

	return mse



def gradMSE(W, b, x, y, reg):
	# ref: https://chunml.github.io/ChunML.github.io/tutorial/Regularization/

	predicted = np.sign([sum(W * x[i] + b) for i in range(len(x))]) # the predicted y

	m = len(predicted) # the amount of data

	for e in range(m): # change -1 to 0
		if predicted[e] == -1: predicted[e] = 0

	# GRADIENT OF WEIGHT
	grad_weight = np.array([])

	for j in range(len(W)): 	# dimension -> 784
		temp = reg * W[j]

		for i in range(m):		# dimension -> 3500
			temp += (predicted[i] - y[i]) * x[i][j]
		temp = temp / m

		grad_weight = np.append(grad_weight, temp)

	# GRADIENT OF BIAS
	grad_bias = sum(predicted - y) / m
	grad_bias = np.full(W.shape[0], grad_bias)	
	
	return grad_weight, grad_bias



# def crossEntropyLoss(W, b, x, y, reg):
#     # Your implementation here

# def gradCE(W, b, x, y, reg):
#     # Your implementation here

def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
	
	for i in range(iterations):		
		grad_W, grad_b = gradMSE(W, b, trainingData, trainingLabels, reg)

		u_W = grad_W / -LA.norm(grad_W)
		u_b = grad_b / -LA.norm(grad_b)

		W += alpha * u_W
		b += alpha * u_b 

		print("iteration: ", i)
		cost = MSE(W, b, trainingData, trainingLabels, reg)
		print("cost: ", cost)

		if cost <= EPS: return W, b
	
	return W, b


# def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
#     # Your implementation here

trainingData, validationData, testingData, trainingTarget, validationTarget, testingTarget = loadData()
trainingData = trainingData.reshape(trainingData.shape[0], trainingData.shape[1] * trainingData.shape[2])
validationData = validationData.reshape(validationData.shape[0], validationData.shape[1] * validationData.shape[2])
testingData = testingData.reshape(testingData.shape[0], testingData.shape[1] * testingData.shape[2])

weight = np.array([1] * trainingData.shape[1], dtype=float)
bias = np.array([1] * trainingData.shape[1], dtype=float)

trainingTarget = trainingTarget.reshape(trainingTarget.shape[0])
validationTarget = validationTarget.reshape(validationTarget.shape[0])
testingTarget = testingTarget.reshape(testingTarget.shape[0])

W, b = grad_descent(weight, bias, trainingData, trainingTarget, 10, 5000, 0, 0.02)

# pre = abs(trainData[0]) < 0.5
# print(pre)
# print(pre.all())

