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
	predicted = np.array([sum(W * x[i]) + b for i in range(len(x))])

	return predicted, len(predicted)  # return the predicted result and number of prediction


def calculate_logistic_prediction(W, b, x):

	# the predicted y
	predicted = np.array([1 / (1 + math.exp(-(sum(W * x[i]) + b))) for i in range(len(x))], dtype=float) 

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
	
	ce = sum([-(y[i] * math.log(predicted[i])) - (1 - y[i]) * math.log(1 - predicted[i]) for i in range(m)]) / m + sum(W ** 2) * reg / 2
	
	return ce


def gradCE(W, b, x, y, reg):
	
	predicted, m = calculate_logistic_prediction(W, b, x)

	# GRADIENT OF WEIGHT
	grad_weight = np.array([reg * W[j] + sum((predicted[i] - y[i]) * x[i][j] for i in range(m)) / m for j in range(len(W))])

	# GRADIENT OF BIAS
	grad_bias = sum(predicted - y) / m

	return grad_weight, grad_bias
	


def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS, lossType="None"):
	
	for i in range(iterations):		

		if lossType == "None": 
			grad_W, grad_b = gradMSE(W, b, trainingData, trainingLabels, reg)
		
		elif lossType == "CE":
			grad_W, grad_b = gradCE(W, b, trainingData, trainingLabels, reg)

		else:
			print("Undefined loss type, please try again")
			return

		W -= alpha * grad_W
		b -= alpha * grad_b
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

		if abs(LA.norm(grad_W)) <= EPS: return W, b

	return W, b


def accuracy(W, b, x, y):
	predicted, m = calculate_binary_prediction(W, b, x)
	
	predicted[predicted <= 0.5] = 0
	predicted[predicted > 0.5] = 1

	predicted = predicted.reshape(predicted.shape[0], 1)

	error_num = sum(sum(abs(predicted - y)))

	accuracy = 1 - error_num / m
	print("accuracy: ", accuracy)
	return accuracy
	


def buildGraph(beta1=0.95, beta2=0.99, epsilon=1e-9, lossType='MSE', learning_rate=0.001, batch_size=500, iterations=700, optimizerType="GD", GraphPlot=False):

	# DATA PROCESSING
	trainingData, validationData, testingData, trainingTarget, validationTarget, testingTarget = loadData()
	
	num_sample = trainingData.shape[0]
	num_pixel = trainingData.shape[1] * trainingData.shape[2]

	trainingData = trainingData.reshape(num_sample, num_pixel)
	trainingTarget = trainingTarget.reshape(num_sample, 1)
	
	validationData = validationData.reshape(validationData.shape[0], num_pixel)
	validationTarget = validationTarget.reshape(validationTarget.shape[0], 1)

	testingData = testingData.reshape(testingData.shape[0], num_pixel)
	testingTarget = testingTarget.reshape(testingTarget.shape[0], 1)

	tf.set_random_seed(421)


	# INITIALIZATION
	x = tf.placeholder(dtype=tf.float32, shape=(batch_size, num_pixel), name="training_data")
	y = tf.placeholder(dtype=tf.float32, shape=(batch_size, 1), name="labels")

	valid_x = tf.placeholder(dtype=tf.float32, shape=(validationData.shape[0], num_pixel), name="validation_data")
	valid_y = tf.placeholder(dtype=tf.float32, shape=(validationTarget.shape[0], 1), name="valid_labels")
	
	testing_x = tf.placeholder(dtype=tf.float32, shape=(testingData.shape[0], num_pixel), name="testing_data")
	testing_y = tf.placeholder(dtype=tf.float32, shape=(testingTarget.shape[0], 1), name="testing_labels")

	weight = tf.Variable(tf.truncated_normal(shape=(num_pixel, 1), stddev=0.5), name="weight")
	bias = tf.Variable(tf.ones((1,1)), name="bias")


	# REGULARIZATION
	reg = tf.placeholder(dtype=tf.float32, name="reg")


	# BUILD MODEL
	predictions = tf.matmul(x, weight) + bias
	predictions_valid = tf.matmul(valid_x, weight) + bias
	predictions_testing = tf.matmul(testing_x, weight) + bias


	if lossType == "MSE":
		loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=y, predictions=predictions))
		loss_valid = tf.reduce_mean(tf.losses.mean_squared_error(labels=valid_y, predictions=predictions_valid))
		loss_testing = tf.reduce_mean(tf.losses.mean_squared_error(labels=testing_y, predictions=predictions_testing))
	
	if lossType == "CE":
		predictions = tf.sigmoid(predictions)
		loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=predictions))
		predictions_valid = tf.sigmoid(predictions_valid)
		loss_valid = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=valid_y, logits=predictions_valid))
		predictions_testing = tf.sigmoid(predictions_testing)
		loss_testing = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=testing_y, logits=predictions_testing))
		

	# REGULARIZATION
	regularizer = tf.nn.l2_loss(weight)
	loss = tf.reduce_mean(loss + reg * regularizer)
	loss_valid = tf.reduce_mean(loss_valid + reg * regularizer)
	loss_testing = tf.reduce_mean(loss_testing + reg * regularizer)

	# OPTIMIZER
	if optimizerType == "GD":
		optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	if optimizerType == "ADAM":
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon).minimize(loss)


	updated_weight, updated_bias = None, None

	train_cost_list, train_acculist, valid_cost_list = [], [], []
	valid_acculist, test_cost_list, test_acculist = [], [], []
	train_cost, valid_cost, test_cost = 0, 0, 0


	# TRAINING START
	with tf.Session() as sess:
	
		sess.run(tf.global_variables_initializer())

		for epoch in range(iterations):
			
			np.random.seed(epoch)
			np.random.shuffle(trainingData)
			np.random.seed(epoch)
			np.random.shuffle(trainingTarget)

			i = 0

			while i < num_sample:

				start = i
				end = i + batch_size

				batch_x = np.array(trainingData[start:end])
				batch_y = np.array(trainingTarget[start:end])

				updated_weight, updated_bias, train_cost, _ = sess.run([weight, bias, loss, optimizer], feed_dict={x: batch_x, y: batch_y, reg: 0})
				
				if GraphPlot: 
					_, _, valid_cost = sess.run([weight, bias, loss_valid], feed_dict={valid_x: validationData, valid_y: validationTarget, reg: 0})

					_, _, test_cost = sess.run([weight, bias, loss_testing], feed_dict={testing_x: testingData, testing_y: testingTarget, reg: 0})
				
				i += batch_size
				
			print("epoch: ",epoch)
			print("train loss(" + lossType + "): ", train_cost)

			if GraphPlot: 
				print("valid loss(" + lossType + "): ", valid_cost)
				print("testing loss(" + lossType + "): ", test_cost)

				updated_weight = updated_weight.flatten()
				updated_bias = updated_bias.flatten()[0]

				train_cost_list.append(train_cost)
				train_acculist.append(accuracy(updated_weight, updated_bias, trainingData, trainingTarget))
				valid_cost_list.append(valid_cost)
				valid_acculist.append(accuracy(updated_weight, updated_bias, validationData, validationTarget))
				test_cost_list.append(test_cost)
				test_acculist.append(accuracy(updated_weight, updated_bias, testingData, testingTarget))

			print(" ")


	if GraphPlot: 
		return train_cost_list, train_acculist, valid_cost_list, valid_acculist, test_cost_list, test_acculist

	# CALCULATE ACCURACY
	else: 
		updated_weight = updated_weight.flatten()
		updated_bias = updated_bias.flatten()[0]

		training_accu = accuracy(updated_weight, updated_bias, validationData, validationTarget)
		valid_accu =	accuracy(updated_weight, updated_bias, validationData, validationTarget)
		testing_accu = accuracy(updated_weight, updated_bias, testingData, testingTarget)

		return training_accu, valid_accu, testing_accu
	# return updated_weight, updated_bias, predictions, y, loss, optimizer, reg

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

	accuracy(W, b, validationData, validationTarget)

# regression_training()



def SGD(lossType, optimizerType):

	print("Optimizer Type is: " + optimizerType)

	train_cost_list, train_acculist, valid_cost_list, valid_acculist, test_cost_list, test_acculist = buildGraph(lossType=lossType, learning_rate=0.001, optimizerType=optimizerType)

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
		


def batch_size_investigation(lossType):

	train_cost_list, train_acculist, valid_cost_list, valid_acculist, test_cost_list, test_acculist = [], [], [], [], [], []

	batch_sizes = [100, 700, 1750]

	for batch_size in batch_sizes:

		train_cost, train_accu, valid_cost, valid_accu, test_cost, test_accu = buildGraph(beta1=0.95, beta2=0.99, epsilon=1e-9, lossType=lossType, learning_rate=0.001, batch_size=batch_size)

		train_cost_list.append(train_cost)
		train_acculist.append(train_accu)
		valid_cost_list.append(valid_cost)
		valid_acculist.append(valid_accu)
		test_cost_list.append(test_cost)
		test_acculist.append(test_accu)
	
	file_name = ['train_cost_list', 'train_accu_list', 'valid_cost_list', 'valid_accu_list', 'test_cost_list', 'test_accu_list']
	li = [train_cost_list, train_acculist, valid_cost_list, valid_acculist, test_cost_list, test_acculist]

	for i in range(len(file_name)):
		for j in range(len(batch_sizes)):
			f = open("./resultLog" + lossType +"/"+file_name[i] + "_" + str(batch_sizes[j]) + ".txt", 'w+')
			f.write(str(li[i][j]))	
			f.close()

	with plt.style.context('Solarize_Light2'):
		
		for i in range(len(batch_sizes)):

			x = np.linspace(0, len(train_cost_list[i]), len(train_cost_list[i]))

			fig, axs = plt.subplots(2, 1)

			axs[0].set_title("Batch Size: " + str(batch_sizes[i]))
			axs[0].plot(x, train_cost_list[i], color='b', label='Training cost')
			axs[0].plot(x, valid_cost_list[i], color='g', label='Validation cost')
			axs[0].plot(x, test_cost_list[i], color='r', label='Testing cost')
			axs[0].set_xlabel('epoch')
			axs[0].set_ylabel('loss')
			axs[0].legend(loc="upper right")
			
			axs[1].plot(x, train_acculist[i], color='b', label='Training accuracy')
			axs[1].plot(x, valid_acculist[i], color='g', label='Validation accuracy')
			axs[1].plot(x, test_acculist[i], color='r', label='Testing accuracy')
			axs[1].set_xlabel('epoch')
			axs[1].set_ylabel('accuracy')
			axs[1].legend(loc="upper right")



		x = np.linspace(0, len(train_cost_list[i]), len(train_cost_list[i]))

		fig, axs = plt.subplots(2, 1)

		axs[0].set_title("Training Set Performance Comparison")
		axs[0].plot(x, train_cost_list[0], color='k', label='Batch Size=' + str(batch_sizes[0]))
		axs[0].plot(x, train_cost_list[1], color='y', label='Batch Size=' + str(batch_sizes[1]))
		axs[0].plot(x, train_cost_list[2], color='m', label='Batch Size=' + str(batch_sizes[2]))
		axs[0].set_xlabel('epoch')
		axs[0].set_ylabel('loss')
		axs[0].legend(loc="upper right")
		
		axs[1].plot(x, train_acculist[0], color='k', label='Batch Size=' + str(batch_sizes[0]))
		axs[1].plot(x, train_acculist[1], color='y', label='Batch Size=' + str(batch_sizes[1]))
		axs[1].plot(x, train_acculist[2], color='m', label='Batch Size=' + str(batch_sizes[2]))
		axs[1].set_xlabel('epoch')
		axs[1].set_ylabel('accuracy')
		axs[1].legend(loc="upper right")



		fig, axs = plt.subplots(2, 1)

		axs[0].set_title("Validation Set Performance Comparison")
		axs[0].plot(x, valid_cost_list[0], color='k', label='Batch Size=' + str(batch_sizes[0]))
		axs[0].plot(x, valid_cost_list[1], color='y', label='Batch Size=' + str(batch_sizes[1]))
		axs[0].plot(x, valid_cost_list[2], color='m', label='Batch Size=' + str(batch_sizes[2]))
		axs[0].set_xlabel('epoch')
		axs[0].set_ylabel('loss')
		axs[0].legend(loc="upper right")
		
		axs[1].plot(x, valid_acculist[0], color='k', label='Batch Size=' + str(batch_sizes[0]))
		axs[1].plot(x, valid_acculist[1], color='y', label='Batch Size=' + str(batch_sizes[1]))
		axs[1].plot(x, valid_acculist[2], color='m', label='Batch Size=' + str(batch_sizes[2]))
		axs[1].set_xlabel('epoch')
		axs[1].set_ylabel('accuracy')
		axs[1].legend(loc="upper right")	


		fig, axs = plt.subplots(2, 1)

		axs[0].set_title("Testing Set Performance Comparison")
		axs[0].plot(x, test_cost_list[0], color='k', label='Batch Size=' + str(batch_sizes[0]))
		axs[0].plot(x, test_cost_list[1], color='y', label='Batch Size=' + str(batch_sizes[1]))
		axs[0].plot(x, test_cost_list[2], color='m', label='Batch Size=' + str(batch_sizes[2]))
		axs[0].set_xlabel('epoch')
		axs[0].set_ylabel('loss')
		axs[0].legend(loc="upper right")
		
		axs[1].plot(x, test_acculist[0], color='k', label='Batch Size=' + str(batch_sizes[0]))
		axs[1].plot(x, test_acculist[1], color='y', label='Batch Size=' + str(batch_sizes[1]))
		axs[1].plot(x, test_acculist[2], color='m', label='Batch Size=' + str(batch_sizes[2]))
		axs[1].set_xlabel('epoch')
		axs[1].set_ylabel('accuracy')
		axs[1].legend(loc="upper right")	

	plt.show()
		



def hyperparameter_investigation(lossType):
	
	beta1 = [0.95, 0.99]
	beta2 = [0.99, 0.9999]
	epsilon = [1e-9, 1e-4]	
	
	inv_graph = []

	for b1 in beta1:
		for b2 in beta2:
			for e in epsilon:

				name = "beta1=" + str(b1) + ", beta2=" + str(b2) + ", epsilon=" + str(e)

				training_accu, valid_accu, testing_accu = buildGraph(beta1=b1, beta2=b2, epsilon=e, lossType=lossType, learning_rate=0.001)
				
				# print(name)
				# print('training accuracy: ', training_accu)
				# print('validation accuracy: ', valid_accu)
				# print('testing accuracy: ', testing_accu)
				inv_graph.append([name, training_accu, valid_accu, testing_accu])
	
	for i in range(len(inv_graph)):
		print(inv_graph[i][0])
		print('training accuracy: ', inv_graph[i][1])
		print('validation accuracy: ', inv_graph[i][2])
		print('testing accuracy: ', inv_graph[i][3])
		print(" ")
				

	# x = np.linspace(0, len(lossTrend), len(lossTrend))

	# with plt.style.context('Solarize_Light2'):
		
	# 	for i in range(len(inv_graph)):
			
	# 		plt.plot(x, inv_graph[i][1], label= inv_graph[i][0] + ", accuracy: " + str(inv_graph[i][2]))

	# 	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
	# 	plt.xlabel('Epoch', fontsize=14)
	# 	plt.ylabel('Loss', fontsize=14)

	plt.show()


	

hyperparameter_investigation('MSE')
# SGD('CE', 'ADAM')
# batch_size_investigation('CE')

# ref: https://chunml.github.io/ChunML.github.io/tutorial/Regularization/
# ref: https://en.wikipedia.org/wiki/Linear_regression
# ref: https://chunml.github.io/ChunML.github.io/tutorial/Regularization/