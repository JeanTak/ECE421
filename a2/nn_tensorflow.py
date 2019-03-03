import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os

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


def conv2d(x, W, b, stride=1):
	conv = tf.nn.conv2d(x, W, strides=[1,stride, stride, 1], padding='SAME')
	conv = tf.nn.bias_add(conv, b)
	return conv

def maxpool2d(x, k=2):  #	     size of window  				movement of window
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def batch_norm_wrapper(inputs, epsilon = 1e-3):
	scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
	beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
	input_mean, input_var = tf.nn.moments(inputs,[0])
	return tf.nn.batch_normalization(inputs, input_mean, input_var, beta, scale, epsilon)

def convolutional_neural_network(x):

	tf.set_random_seed(421)
	
	weights = {'w_conv': tf.get_variable("weight", shape=(4, 4, 1, 32), initializer=tf.contrib.layers.xavier_initializer()),					 
	}
	
	biases = {'b_conv': tf.get_variable("bias", shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
	}

	conv = conv2d(x, weights['w_conv'], biases['b_conv'])
	conv_relu = tf.nn.relu(conv)
	batch_norm = batch_norm_wrapper(conv_relu)
	max_pool = maxpool2d(batch_norm)
	flatten_l = tf.contrib.layers.flatten(max_pool)
	fc1 = tf.contrib.layers.fully_connected(flatten_l, 784)
	output = tf.contrib.layers.fully_connected(fc1, 10, activation_fn=tf.nn.softmax)
	
	return output


def train_neural_network(batch_size=32, n_classes=10, iterations=50):

	trainingData, validationData, testingData, trainingTarget, validationTarget, testingTarget = loadData()
	trainingTarget, validationTarget, testingTarget = convertOneHot(trainingTarget, validationTarget, testingTarget)
	
	trainingData = np.reshape(trainingData, (-1, 28, 28, 1))

	num_sample = trainingData.shape[0]

	x = tf.placeholder(dtype=tf.float32, shape=(batch_size, trainingData.shape[1] * trainingData.shape[2]), name="training_data")
	y = tf.placeholder(dtype=tf.float32, shape=(batch_size, n_classes), name="labels")

	x = tf.reshape(x, shape=[-1, 28, 28, 1])

	predictions = convolutional_neural_network(x)
	CE_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=predictions))
	optimizer = tf.train.AdamOptimizer().minimize(CE_loss)

		# TRAINING START
	with tf.Session() as sess:
		
		print("Amount of Samples: ", num_sample)
		sess.run(tf.global_variables_initializer())

		for epoch in range(iterations):
			
			np.random.seed(epoch)
			np.random.shuffle(trainingData)
			np.random.seed(epoch)
			np.random.shuffle(trainingTarget)

			epoch_loss, i = 0, 0
			while i < num_sample:

				start = i
				end = i + batch_size

				batch_x = np.array(trainingData[start:end])
				batch_y = np.array(trainingTarget[start:end])

				_, cost = sess.run([optimizer, CE_loss], feed_dict={x: batch_x, y: batch_y})
				epoch_loss += cost
				print("Batch: ", i, ", Batch Loss: ", cost)
				
				
				i += batch_size
				
			print("epoch: ",epoch)
			print("train loss(CE): ", epoch_loss)


# trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
# trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget)

train_neural_network()

