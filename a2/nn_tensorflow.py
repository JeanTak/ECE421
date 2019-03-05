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
        trainData, trainTarget = Data[:15000], Target[:15000]
        validData, validTarget = Data[15000:16000], Target[15000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

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
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')

def batch_norm_wrapper(inputs, epsilon = 0.0000001):
	mean, variance = tf.nn.moments(inputs, [0])
	return tf.nn.batch_normalization(inputs, mean, variance, None, None, epsilon, name=None)

# def flatten_layer(layer):
# 	layer_shape = layer.get_shape()
# 	num_features = layer_shape[1:4].num_elements()
# 	layer = tf.reshape(layer, [-1, num_features])
# 	return layer

def convolutional_neural_network(x, mode='training'):

	tf.set_random_seed(421)
	
	weights = {'w_conv': tf.get_variable("conv_weight_" + mode, shape=(3, 3, 1, 32), initializer=tf.contrib.layers.xavier_initializer()),
		         'w_fc1': tf.get_variable("fc1_weight_" + mode, shape=(6272, 784), initializer=tf.contrib.layers.xavier_initializer()),
						 'w_fc2': tf.get_variable("fc2_weight_" + mode, shape=(784, 10), initializer=tf.contrib.layers.xavier_initializer()),
	}
	
	biases = {'b_conv': tf.get_variable("conv_bias_" + mode, shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
					  'b_fc1': tf.get_variable("fc1_bias_" + mode, shape=(784), initializer=tf.contrib.layers.xavier_initializer()),
						'b_fc2': tf.get_variable("fc2_bias_" + mode, shape=(10), initializer=tf.contrib.layers.xavier_initializer()),
	}

	conv = conv2d(x, weights['w_conv'], biases['b_conv'])
	conv_relu = tf.nn.relu(conv)
	batch_norm = batch_norm_wrapper(conv_relu)
	max_pool = maxpool2d(batch_norm)

	# flatten_l = flatten_layer(max_pool)
	flatten_l = tf.layers.flatten(max_pool)

	fc1 = tf.add(tf.matmul(flatten_l, weights['w_fc1']), biases['b_fc1'], name='logit')
	fc1_relu = tf.nn.relu(fc1)
	fc2 = tf.add(tf.matmul(fc1_relu, weights['w_fc2']), biases['b_fc2'], name='logit')
	# output = tf.nn.softmax(fc2)
	# return output
	return fc2


def train_neural_network(batch_size=32, n_classes=10, iterations=50):

	trainingData, validationData, testingData, trainingTarget, validationTarget, testingTarget = loadData()
	trainingTarget, validationTarget, testingTarget = convertOneHot(trainingTarget, validationTarget, testingTarget)
	
	trainingData = np.reshape(trainingData, (-1, 28, 28, 1))
	validationData = np.reshape(validationData, (-1, 28, 28, 1))

	num_sample = trainingData.shape[0]

	# x = tf.placeholder(dtype=tf.float32, shape=(batch_size, 28, 28, 1), name="training_data")
	# y = tf.placeholder(dtype=tf.float32, shape=(batch_size, n_classes), name="training_labels")
	x = tf.placeholder(dtype=tf.float32, name="data")
	y = tf.placeholder(dtype=tf.float32, name="predictions")

	# x_valid = tf.placeholder(dtype=tf.float32, shape=(validationData.shape[0], 28, 28, 1), name="valid_data")
	# y_valid = tf.placeholder(dtype=tf.float32, shape=(validationData.shape[0], n_classes), name="valid_labels")

	predictions = convolutional_neural_network(x)
	CE_loss = tf.losses.softmax_cross_entropy(y, predictions)
	optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(CE_loss)

	# predictions_validation = convolutional_neural_network(x_valid, mode="validation")
	correct = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

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
			while i + batch_size < num_sample:

				start = i
				end = i + batch_size

				batch_x = np.array(trainingData[start:end])
				batch_y = np.array(trainingTarget[start:end])

				_, cost = sess.run([optimizer, CE_loss], feed_dict={x: batch_x, y: batch_y})
				epoch_loss += cost
				# print("Batch: ", i, ", Batch Loss: ", cost)
				
				
				i += batch_size
				
			print("epoch: ",epoch)
			print("train loss(CE): ", epoch_loss)

			print('Accuracy:', sess.run(accuracy, feed_dict={x: validationData, y: validationTarget}))



train_neural_network()

# https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-tensorflow/
# https://www.kaggle.com/pouryaayria/convolutional-neural-networks-tutorial-tensorflow