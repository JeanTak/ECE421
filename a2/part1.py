import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
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
	return np.maximum(x, 0)

def softmax(x): 
    x_stable = (x.transpose() - np.max(x, axis=1).reshape(1, -1)).transpose()
    return (np.exp(x_stable).transpose() / np.sum(np.exp(x_stable), axis=1).reshape(1, -1)).transpose()

def computeLayer(X, W, b):
	return np.dot(X,W) + b

def CE(target, prediction):
    score = softmax(prediction)
    ce = -np.mean(np.sum(target * np.log(score), axis=1))
    return ce

def gradCE(target, prediction):
    score = softmax(prediction)
    return (-1) * target * (1. / score)



# 1.2 BACKPROPAGATION DERIVATION

def grad_relu(x):
    return 1. * (x > 0)

def loss_accuracy(w_hidden, b_hidden, w_outer, b_outer, x, target):
    s_hidden = computeLayer(x, w_hidden, b_hidden)
    y_hidden = relu(s_hidden)
    s_outer = computeLayer(y_hidden, w_outer, b_outer)
    y_outer = softmax(s_outer)
    loss = CE(target, s_outer)
    accuracy = (np.sum(np.argmax(target, axis=1) == np.argmax(y_outer, axis=1)) / len(target) * 100)
    return loss, accuracy

def train(trainData, validData, testData, trainTarget, testTarget, validTarget, units, alpha, epoch):
    # Hidden Layer Data Initialization
    s_hidden, y_hidden = np.zeros(0), np.zeros(0)
    w_hidden = np.random.normal(0, np.sqrt(2 / (trainData.shape[1] + units)), units * trainData.shape[1]).reshape(trainData.shape[1], units)
    b_hidden = np.zeros(units).reshape(1, -1)
    # Output Layer Initializationn
    s_outer, y_outer = np.zeros(0), np.zeros(0) # z = s_outer, p = y_outer, y = newtrain
    sigma = np.sqrt(2 / (10 + units))
    w_outer = np.random.normal(0, sigma, 10 * units).reshape(units, 10)
    b_outer = np.zeros(10).reshape(1, -1)
    # Momentum Initialization
    mmt_coeff = 0.9
    v_w_hidden = np.ones((trainData.shape[1], units)) * 1e-5
    v_w_outer = np.ones((units, 10)) * 1e-5
    v_b_hidden = np.ones((1, units)) * 1e-5
    v_b_outer = np.ones((1, 10)) * 1e-5
    # Plotting
    trainloss, validloss, testloss = [], [], []
    trainAccuracy, validAccuracy, testAccuracy = [], [], []
    
    for i in range(epoch):
        print("iteration: ", i)
        # Forward propagation
        s_hidden = computeLayer(trainData, w_hidden, b_hidden)
        y_hidden = relu(s_hidden)
        s_outer = computeLayer(y_hidden, w_outer, b_outer)
        y_outer = softmax(s_outer)
        # Back Propagation
        dLdsum = y_outer - trainTarget
        
        grad_w_outer = np.dot(dLdsum.T, y_hidden).T # shape is correct        
        grad_b_outer = np.sum(dLdsum, axis=0)
        temp = np.dot(w_outer, dLdsum.T) * grad_relu(s_hidden).T
        grad_w_hidden = np.dot(trainData.T, temp.T)
        grad_b_hidden = np.sum(np.dot(dLdsum, (w_outer.T)) * grad_relu(s_hidden), axis=0)
        
        # Update momentum term
        v_w_hidden = mmt_coeff * v_w_hidden + alpha * grad_w_hidden / trainData.shape[0]
        v_w_outer = mmt_coeff * v_w_outer + alpha * grad_w_outer / trainData.shape[0]
        v_b_hidden = mmt_coeff * v_b_hidden +alpha * grad_b_hidden / trainData.shape[0]
        v_b_outer = mmt_coeff * v_b_outer + alpha * grad_b_outer / trainData.shape[0]
        # Update weight and bias matrices
        w_hidden -= v_w_hidden
        b_hidden -= v_b_hidden
        w_outer -= v_w_outer
        b_outer -= v_b_outer
        
        trainLoss, trainaccuracy = loss_accuracy(w_hidden, b_hidden, w_outer, b_outer, trainData, trainTarget)
        validLoss, validaccuracy = loss_accuracy(w_hidden, b_hidden, w_outer, b_outer, validData, validTarget)
        testLoss, testaccuracy = loss_accuracy(w_hidden, b_hidden, w_outer, b_outer, testData, testTarget)
        
        trainloss.append(trainLoss)
        validloss.append(validLoss)
        testloss.append(testLoss)
        trainAccuracy.append(trainaccuracy)
        validAccuracy.append(validaccuracy)
        testAccuracy.append(testaccuracy)
        #if i == 199:
        print("Train Accuracy: ", trainaccuracy)
        print("Valid Accuracy: ", validaccuracy)
        print("Test Accuracy: ", testaccuracy)
    # Plotting
    
    plt.figure(num = None, figsize = (10, 5), dpi = 150, facecolor = 'w', edgecolor = 'k')
    plt.subplot(1, 2, 1)
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.margins(0.05)
    plt.plot(range(0, len(trainloss)), trainloss, label='train Loss')
    plt.plot(range(0, len(validloss)), validloss, label='train Loss')
    plt.plot(range(0, len(testloss)), testloss, label='train Loss')
    plt.legend(("train", "valid", "test"))
    #print("Train Loss: ", trainLoss)
    #print("Valid Loss: ", validLoss)
    #print("Test Loss: ", testLoss)
    
    
    plt.subplot(1, 2, 2)
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy %')
    plt.grid(True)
    plt.margins(0.05)
    plt.plot(range(0, len(trainAccuracy)), trainAccuracy, label='train accuracy')
    plt.plot(range(0, len(validAccuracy)), validAccuracy, label='valid accuracy')
    plt.plot(range(0, len(testAccuracy)), testAccuracy, label='test accuracy')
        
    plt.legend(("train", "valid", "test"))
        
    return w_hidden, b_hidden, w_outer, b_outer


trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
newtrain, newvalid, newtest = convertOneHot(trainTarget, validTarget, testTarget)
trainingData = trainData.reshape(trainData.shape[0], -1)
validData = validData.reshape(validData.shape[0], -1)
testData = testData.reshape(testData.shape[0], -1)
start = time.time()
w_hidden, b_hidden, w_outer, b_outer = train(trainingData, validData, testData, newtrain, newtest, newvalid, units=100, alpha=0.1, epoch=200)
end = time.time()
print("time:", end - start)
# ref: https://deepnotes.io/softmax-crossentropy
# ref: https://hackernoon.com/how-to-initialize-weights-in-a-neural-net-so-it-performs-well-3e9302d4490f
# ref: https://medium.com/@aerinykim/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d