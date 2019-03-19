import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target




# 1.1 HELPER FUNCTIONS
def relu(x):
	return max(x, 0)

def softmax(x):
	exp_x = np.exp(x)
	return np.divide(exp_x, np.sum(exp_x))


def computeLayer(X, W, b):
	predict = np.dot(X, W) + b
	return predict

def CE(target, prediction):
	score = softmax(prediction)
	ce = np.dot(np.transpose(target), np.log(score))
	return ce / len(target)

def gradCE(target, prediction):
	grad_ce = np.subtract(target, prediction)
	return grad_ce / len(target)



# 1.2 BACKPROPAGATION DERIVATION

def grad_relu(x):
	return max(np.sign(x), 0)

def grad_outer_weight_bias(target, prediction, activations):
	delta = np.multiply(gradCE(target, prediction), grad_relu(target))
	grad_w = np.multiply(delta, activations[-1].transpose())
	grad_b = np.sum(delta)
	return grad_w, grad_b, delta

# def grad_hidden_weight_bias(prediction, delta_, activations, weights, l):
	# delta_cur = np.dot(weights[-l+1].transpose(), delta_)

# ref: https://deepnotes.io/softmax-crossentropy