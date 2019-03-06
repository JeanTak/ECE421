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


def convolutional_neural_network(x):

	tf.set_random_seed(421)
	
	weights = {'w_conv': tf.get_variable("w_conv", shape=(3, 3, 1, 32), initializer=tf.contrib.layers.xavier_initializer()),
		         'w_fc1': tf.get_variable("w_fc1", shape=(6272, 784), initializer=tf.contrib.layers.xavier_initializer()),
						 'w_fc2': tf.get_variable("w_fc2", shape=(784, 10), initializer=tf.contrib.layers.xavier_initializer()),
	}
	
	biases = {'b_conv': tf.get_variable("b_conv", shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
					  'b_fc1': tf.get_variable("b_fc1", shape=(784), initializer=tf.contrib.layers.xavier_initializer()),
						'b_fc2': tf.get_variable("b_fc2", shape=(10), initializer=tf.contrib.layers.xavier_initializer()),
	}

	# 3 x 3 convolutional layer, with 32 lters, using vertical and horizontal strides of 1
	conv = tf.nn.conv2d(x, weights['w_conv'], strides=[1, 1, 1, 1], padding='SAME')
	conv = tf.nn.bias_add(conv, biases['b_conv'])

	# ReLU activation
	conv_relu = tf.nn.relu(conv)

	# batch normalization layer
	mean, variance = tf.nn.moments(conv_relu, [0])
	batch_norm = tf.nn.batch_normalization(conv_relu, mean, variance, None, None, 0.0000001, name=None)

	# 2 x 2 max pooling layer
	max_pool = tf.nn.max_pool(batch_norm, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

	# Flatten layer
	flatten_l = tf.layers.flatten(max_pool)

	# Fully connected layer (with 784 output units, i.e. corresponding to each pixel)
	fc1 = tf.add(tf.matmul(flatten_l, weights['w_fc1']), biases['b_fc1'], name='logit')

	#ReLU activation
	fc1_relu = tf.nn.relu(fc1)

	# Fully connected layer (with 10 output units, i.e. corresponding to each class)
	fc2 = tf.add(tf.matmul(fc1_relu, weights['w_fc2']), biases['b_fc2'], name='logit2')

	return fc2, weights


def train_neural_network(batch_size=32, n_classes=10, iterations=50, learning_rate=0.0001, reg=0.01, is_reg=False, only_final_accu=False):

	# Load Data
	trainingData, validationData, testingData, trainingTarget, validationTarget, testingTarget = loadData()

	# One Hot
	trainingTarget, validationTarget, testingTarget = convertOneHot(trainingTarget, validationTarget, testingTarget)
	
	# Reshape the datasets to fit the convolutional layer
	trainingData = np.reshape(trainingData, (-1, 28, 28, 1))
	validationData = np.reshape(validationData, (-1, 28, 28, 1))
	testingData = np.reshape(testingData, (-1, 28, 28, 1))

	num_training_sample = trainingData.shape[0]

	x = tf.placeholder(dtype=tf.float32, name="data")
	y = tf.placeholder(dtype=tf.float32, name="predictions")

	predictions, weights = convolutional_neural_network(x)

	# Softmax output + Cross Entropy loss
	CE_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(y, predictions))

	# Regularization
	if is_reg:
		print('Regularization decay coefficient: ', reg)
		regularizer0 = tf.nn.l2_loss(weights['w_conv'])
		regularizer1 = tf.nn.l2_loss(weights['w_fc1'])
		regularizer2 = tf.nn.l2_loss(weights['w_fc2'])
		CE_loss = tf.reduce_mean(CE_loss + reg * regularizer0 + reg * regularizer1 + reg * regularizer2)

	# Adam Optmizer
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(CE_loss)

	correct = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

	train_cost_list, valid_cost_list, test_cost_list = [], [], []
	train_acculist, valid_acculist, test_acculist = [], [], []

	# TRAINING START
	with tf.Session() as sess:
		
		print("Amount of Samples: ", num_training_sample)

		sess.run(tf.global_variables_initializer())

		for epoch in range(iterations):
			
			np.random.seed(epoch)
			np.random.shuffle(trainingData)
			np.random.seed(epoch)
			np.random.shuffle(trainingTarget)

			epoch_loss, i = 0, 0

			while i < num_training_sample:

				start, end = i, i + batch_size

				_, cost = sess.run([optimizer, CE_loss], feed_dict={x: trainingData[start:end], y: trainingTarget[start:end]})
				epoch_loss += cost
				i += batch_size
				# print("Batch: ", i, ", Batch Loss: ", cost)

			print("epoch: ",epoch)
			if not only_final_accu:
				accu_training, loss_training = sess.run([accuracy, CE_loss], feed_dict={x: trainingData, y: trainingTarget})
				accu_valid, loss_valid = sess.run([accuracy, CE_loss], feed_dict={x: validationData, y: validationTarget})
				accu_testing, loss_testing = sess.run([accuracy, CE_loss], feed_dict={x: testingData, y: testingTarget})
			
				train_cost_list.append(loss_training)
				valid_cost_list.append(loss_valid)
				test_cost_list.append(loss_testing)

				train_acculist.append(accu_training)
				valid_acculist.append(accu_valid)
				test_acculist.append(accu_testing)

				print("train loss(CE): ", loss_training)
				print("validation loss(CE): ", loss_valid)
				print("testing loss(CE): ", loss_testing)
				print('Training Accuracy:', accu_training)
				print('Validation Accuracy:', accu_valid)
				print('Testing Accuracy:', accu_testing)

		if only_final_accu:
			accu_training, loss_training = sess.run([accuracy, CE_loss], feed_dict={x: trainingData, y: trainingTarget})
			accu_valid, loss_valid = sess.run([accuracy, CE_loss], feed_dict={x: validationData, y: validationTarget})
			accu_testing, loss_testing = sess.run([accuracy, CE_loss], feed_dict={x: testingData, y: testingTarget})
		
			train_cost_list.append(loss_training)
			valid_cost_list.append(loss_valid)
			test_cost_list.append(loss_testing)

			train_acculist.append(accu_training)
			valid_acculist.append(accu_valid)
			test_acculist.append(accu_testing)

			print("final train loss(CE): ", loss_training)
			print("final validation loss(CE): ", loss_valid)
			print("final testing loss(CE): ", loss_testing)
			print('final training Accuracy:', accu_training)
			print('final validation Accuracy:', accu_valid)
			print('final testing Accuracy:', accu_testing)


	return train_cost_list, valid_cost_list, test_cost_list, train_acculist, valid_acculist, test_acculist


def model_training():
	
	train_cost_list, valid_cost_list, test_cost_list, train_acculist, valid_acculist, test_acculist = train_neural_network(iterations=50)

	# Plot Graph
	x = np.linspace(0, len(train_cost_list), len(train_cost_list))

	with plt.style.context('Solarize_Light2'):
		fig, axs = plt.subplots(2, 1)

		plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
		axs[0].plot(x, train_cost_list, color='b', label='Training cost')
		axs[0].plot(x, valid_cost_list, color='g', label='Validation cost')
		axs[0].plot(x, test_cost_list, color='r', label='Testing cost')
		axs[0].set_xlabel('epoch')
		axs[0].set_ylabel('loss')
		axs[0].legend(loc="upper right")
		
		axs[1].plot(x, train_acculist, color='b', label='Training accuracy')
		axs[1].plot(x, valid_acculist, color='g', label='Validation accuracy')
		axs[1].plot(x, test_acculist, color='r', label='Testing accuracy')
		axs[1].set_xlabel('epoch')
		axs[1].set_ylabel('accuracy')
		axs[1].legend(loc="upper right")

	plt.show()

def L2_Normalization():
	train_cost_list, valid_cost_list, test_cost_list, train_acculist, valid_acculist, test_acculist = train_neural_network(is_reg=True, only_final_accu=True, reg=0.5)

L2_Normalization()
# model_training()


# train_neural_network()

# https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-tensorflow/
# https://www.kaggle.com/pouryaayria/convolutional-neural-networks-tutorial-tensorflow