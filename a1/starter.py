import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math

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



def calculate_binary_prediction(W, b, x):
	
	# the predicted y
	# predicted = np.sign([sum(W * x[i]) + b for i in range(len(x))]) 

	# predicted[predicted < 0] = 0

	predicted = np.array([sum(W * x[i]) + b for i in range(len(x))])

	return predicted, len(predicted)  # return the predicted result and number of prediction


def calculate_logistic_prediction(W, b, x):

	# the predicted y
	predicted = np.array([1 / (1 + math.exp(-(sum(W * x[i]) + b))) for i in range(len(x))], dtype=float) 

	predicted[predicted < 0] = 0

	return predicted, len(predicted)  # return the predicted result and number of prediction


def MSE(W, b, x, y, reg):

	predicted, m = calculate_binary_prediction(W, b, x)

	# calculate regularized MSE
	mse = sum((predicted - y) ** 2) / (2 * m) + sum(W ** 2) * reg / 2

	return mse


def gradMSE(W, b, x, y, reg):

	predicted, m = calculate_binary_prediction(W, b, x)

	# GRADIENT OF WEIGHT
	grad_weight = np.array([reg * W[j] + sum((predicted[i] - y[i]) * x[i][j] for i in range(m)) / m for j in range(len(W))])
	# GRADIENT OF BIAS
	grad_bias = sum(predicted - y) / m
	
	return grad_weight, grad_bias


def crossEntropyLoss(W, b, x, y, reg):

	predicted, m = calculate_logistic_prediction(W, b, x)
	# ce = sum([-(y[i] * math.log(predicted[i])) - (1 - y[i]) * math.log(1 - predicted[i]) for i in range(m)]) / m + sum(W ** 2) * reg / (2 * m)
	ce = sum([-(y[i] * math.log(predicted[i])) - (1 - y[i]) * math.log(1 - predicted[i]) for i in range(m)]) / m + sum(W ** 2) * reg / 2

	return ce


def gradCE(W, b, x, y, reg):
	
	predicted, m = calculate_logistic_prediction(W, b, x)

	# GRADIENT OF WEIGHT
	# grad_weight = np.array([reg * W[j] / m + sum((predicted[i] - y[i]) * x[i][j] for i in range(m)) / m for j in range(len(W))])
	grad_weight = np.array([reg * W[j] + sum((predicted[i] - y[i]) * x[i][j] for i in range(m)) / m for j in range(len(W))])

	# GRADIENT OF BIAS
	grad_bias = sum(predicted - y) / m

	return grad_weight, grad_bias
	


def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS, lossType="None"):
	
	prev_diff = 0 
	for i in range(iterations):		

		if lossType == "None": 
			grad_W, grad_b = gradMSE(W, b, trainingData, trainingLabels, reg)
		
		elif lossType == "CE":
			grad_W, grad_b = gradCE(W, b, trainingData, trainingLabels, reg)

		else:
			print("Undefined loss type, please try again")
			return

		# u_W = grad_W / -LA.norm(grad_W)
		# u_b = grad_b / -LA.norm(grad_b)

		# W += alpha * u_W
		# b += alpha * u_b * grad_b 
		W -= alpha * grad_W
		b -= alpha + grad_b
		print("iteration: ", i)
		print("weight[0]: ", W[0])
		
		if lossType == "None": 		
			cost = MSE(W, b, trainingData, trainingLabels, reg)
			print("loss type: MSE")
			
		elif lossType == "CE":
			cost = crossEntropyLoss(W, b, trainingData, trainingLabels, reg)
			print("loss type: CE")

		print("cost: ", cost)
		print(" ")

		# if cost <= EPS: return W, b
		if abs(prev_diff - LA.norm(W)) <= EPS: return W, b
		prev_diff = LA.norm(W)

	return W, b


def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):

	trainingData, validationData, testingData, trainingTarget, validationTarget, testingTarget = loadData()

	trainingData = trainingData.reshape(trainingData.shape[0], trainingData.shape[1] * trainingData.shape[2])
	validationData = validationData.reshape(validationData.shape[0], validationData.shape[1] * validationData.shape[2])
	testingData = testingData.reshape(testingData.shape[0], testingData.shape[1] * testingData.shape[2])

	trainingTarget = trainingTarget.reshape(trainingTarget.shape[0])
	validationTarget = validationTarget.reshape(validationTarget.shape[0])
	testingTarget = testingTarget.reshape(testingTarget.shape[0])

	x = tf.placeholder('float')
	y = tf.placeholder('float')

	weight = tf.Variable(tf.truncated_normal(shape=trainingData.shape[1], stddev=0.5))
	bias = tf.Variable(tf.truncated_normal(shape=1, stddev=0.5))


	# bias = tf.Variable(tf.random.truncated_normal(shape=1, stddev=0.5))
	

	# loss = tf.losses.sigmoid_cross_entropy()



def accuracy(W, b, x, y):
	predicted, m = calculate_binary_prediction(W, b, x)
	
	predicted[predicted <= 0.5] = 0
	predicted[predicted > 0.5] = 1

	error_num = sum(abs(predicted - y))

	accuracy = 1 - error_num / m
	print("accuracy: ", accuracy)
	return accuracy
	

def regression_training():
	trainingData, validationData, testingData, trainingTarget, validationTarget, testingTarget = loadData()

	trainingData = trainingData.reshape(trainingData.shape[0], trainingData.shape[1] * trainingData.shape[2])
	validationData = validationData.reshape(validationData.shape[0], validationData.shape[1] * validationData.shape[2])
	testingData = testingData.reshape(testingData.shape[0], testingData.shape[1] * testingData.shape[2])

	trainingTarget = trainingTarget.reshape(trainingTarget.shape[0])
	validationTarget = validationTarget.reshape(validationTarget.shape[0])
	testingTarget = testingTarget.reshape(testingTarget.shape[0])


	# # LINEAR
	# weight = np.array([0] * trainingData.shape[1], dtype=float)
	# bias = 1
	# W, b = grad_descent(weight, bias, trainingData, trainingTarget, alpha=0.005, iterations=5000, reg=0, EPS=0, lossType="None")

	# LOGISTIC
	weight = np.array([0.0001] * trainingData.shape[1], dtype=float)
	bias = 1
	W, b = grad_descent(weight, bias, trainingData, trainingTarget, alpha=0.05, iterations=5000, reg=0.1, EPS=0.00000001, lossType="CE")

	accuracies = accuracy(W, b, validationData, validationTarget)


regression_training()

# ref: https://chunml.github.io/ChunML.github.io/tutorial/Regularization/
# ref: https://en.wikipedia.org/wiki/Linear_regression
# ref: https://chunml.github.io/ChunML.github.io/tutorial/Regularization/