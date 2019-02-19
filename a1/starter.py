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


def classifier_MSE(W, b, x): 
	predict = np.dot(x, W) + b
	return predict

def MSE(W, b, x, y, reg):
	predict = classifier_MSE(W, b, x)
	mse = np.mean(np.square(predict - y)) / 2 + 0.5 * reg * np.matmul(np.transpose(W), W)
	return mse


def gradMSE(W, b, x, y, reg): 
	predicted = classifier_MSE(W, b, x)
	gradient_w = np.matmul(np.transpose(x), (predicted - y))
	gradient_w /= len(x)
	gradient_w += reg * W
	gradient_b = np.sum(predicted - y) / len(x)
	return gradient_w, gradient_b


def crossEntropyLoss(W, b, x, y, reg):
	predict = 1 / (1 + np.exp(-(np.dot(x, W) + b)))
	ce = -(np.dot(np.transpose(y), (np.log(predict))) + (1.0 - y).T.dot(np.log(1 - predict))) / len(x)
	ce += reg* 0.5 * np.matmul(np.transpose(W), W)
	return ce


def gradCE(W, b, x, y, reg):    
	predict = 1 / (1 + np.exp(-(np.dot(x, W) + b)))
	grad_w = np.dot(np.transpose(x), (predict - y)) / len(x)
	grad_w += reg * W
	grad_b = np.sum(predict - y) / len(x)
	return grad_w, grad_b


def grad_descent(W, b, x, y, testData, testTarget, validData, validTarget, alpha, iterations, reg, EPS, lossType="MSE"):
  
	trainingloss, validloss, testloss = [], [], []
	trainA, validA, testA = [], [], []
	
	for i in range(iterations):

		if lossType == "MSE":
			grad_weight, grad_bias = gradMSE(W, b, x, y, reg)
		elif lossType == "CE":
			grad_weight, grad_bias = gradCE(W, b, x, y, reg)
		else:
			return

		W -= alpha * grad_weight
		b -= alpha * grad_bias
		
		trainA.append(accuracy(W, b, x, y))
		validA.append(accuracy(W, b, validData, validTarget))
		testA.append(accuracy(W, b, testData, testTarget))
		
		if lossType == "MSE":
			error = MSE(W,b,x,y,reg)[0][0]
			valid_loss = MSE(W, b, validData, validTarget, reg)[0][0]
			testing_loss = MSE(W, b, testData, testTarget, reg)[0][0]
			trainingloss.append(error)
			validloss.append(valid_loss)
			testloss.append(testing_loss)
		
		elif lossType == "CE":
			error = crossEntropyLoss(W,b,x,y,reg)[0][0]
			valid_loss = crossEntropyLoss(W, b, validData, validTarget, reg)[0][0]
			testing_loss = crossEntropyLoss(W, b, testData, testTarget, reg)[0][0]
			trainingloss.append(error)	
			validloss.append(valid_loss)
			testloss.append(testing_loss)
			
		difference = LA.norm(alpha * grad_weight)
		
		print("differennce", difference)
		print("iteraion:", i)
		print("loss:", error)
		print("test loss:", testing_loss)
		print("test loss:", valid_loss)
		
		if difference <= EPS:
			return W, b
		
	plt.title('MSE: epoch = 5000, alpha = 0.005')    
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.margins(0.05)
	plt.plot(range(0, len(testloss)), testloss, label='Test Loss')
	plt.plot(range(0, len(trainingloss)), trainingloss, label='Training Loss')
	plt.plot(range(0, len(validloss)), validloss, label='Validation Loss')

	# plt.plot(range(0, len(testA)), testA, label='Test Accuracy')
	# plt.plot(range(0, len(trainA)), trainA, label='Train Accuracy')
	# plt.plot(range(0, len(validA)), validA, label='Valid Accuracy')
	
	plt.legend(("test", "train", "validation"))
	plt.grid(True)
	
	# plt.autoscale(tight=True)
	# plt.show()
	
	return W,b


def accuracy(W, b, x, y):
	predicted = np.dot(x, W) + b
	
	predicted[predicted <= 0.5] = 0
	predicted[predicted > 0.5] = 1
	
	error_num = np.sum(abs(predicted - y))   
	accuracy = 1 - error_num / len(predicted)
	print("accuracy: ", accuracy)
	return accuracy


def normalEquation(x, y):
	first = np.linalg.inv(np.matmul(np.transpose(x), x))
	return np.matmul(np.matmul(first, np.transpose(x)), y)


def buildGraph(beta1=None, beta2=None, epsilon=None, lossType='MSE', learning_rate=0.001, batch_size=500, iterations=700, optimizerType="GD", GraphPlot=False):

	# DATA PROCESSING
	trainingData, validationData, testingData, trainingTarget, validationTarget, testingTarget = loadData()
	
	num_sample = trainingData.shape[0]
	num_pixel = trainingData.shape[1] * trainingData.shape[2]

	trainingData = trainingData.reshape(num_sample, num_pixel)
	validationData = validationData.reshape(validationData.shape[0], num_pixel)
	testingData = testingData.reshape(testingData.shape[0], num_pixel)

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
		loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=predictions))
		loss_valid = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=valid_y, logits=predictions_valid))
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
		if beta1 != None:
			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(loss)
		elif beta2 != None:
			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta2=beta2).minimize(loss)
		elif epsilon != None: 
			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon).minimize(loss)
		else:
			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


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

				updated_bias = updated_bias.flatten()[0]
				
				train_cost_list.append(train_cost)
				train_acculist.append(accuracy(updated_weight, updated_bias, trainingData, trainingTarget))
				valid_cost_list.append(valid_cost)
				valid_acculist.append(accuracy(updated_weight, updated_bias, validationData, validationTarget))
				test_cost_list.append(test_cost)
				test_acculist.append(accuracy(updated_weight, updated_bias, testingData, testingTarget))

			print(" ")


	if GraphPlot: 
		return train_cost_list, train_acculist, valid_cost_list, valid_acculist, test_cost_list, test_acculist, updated_weight, updated_bias, predictions, y, loss, optimizer, reg

	# CALCULATE ACCURACY
	else: 
		updated_bias = updated_bias.flatten()[0]

		training_accu = accuracy(updated_weight, updated_bias, trainingData, trainingTarget)
		valid_accu =	accuracy(updated_weight, updated_bias, validationData, validationTarget)
		testing_accu = accuracy(updated_weight, updated_bias, testingData, testingTarget)

		return training_accu, valid_accu, testing_accu, updated_weight, updated_bias, predictions, y, loss, optimizer, reg



def SGD(lossType, optimizerType):

	print("Optimizer Type is: " + optimizerType)

	train_cost_list, train_acculist, valid_cost_list, valid_acculist, test_cost_list, test_acculist, _, _, _, _, _, _, _ = buildGraph(lossType=lossType, learning_rate=0.001, optimizerType=optimizerType, GraphPlot=True)

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



def batch_size_investigation(lossType):

	train_cost_list, train_acculist, valid_cost_list, valid_acculist, test_cost_list, test_acculist = [], [], [], [], [], []

	batch_sizes = [100, 700, 1750]

	for batch_size in batch_sizes:

		train_cost, train_accu, valid_cost, valid_accu, test_cost, test_accu, _, _, _, _, _, _, _ = buildGraph(lossType=lossType, learning_rate=0.001, batch_size=batch_size, optimizerType='ADAM', GraphPlot=True)

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

	# Plot Graph
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
		name = "beta1=" + str(b1) + ", beta2=0, epsilon=0"
		training_accu, valid_accu, testing_accu, _, _, _, _, _, _, _ = buildGraph(beta1=b1, lossType=lossType, learning_rate=0.001, optimizerType='ADAM')
		inv_graph.append([name, training_accu, valid_accu, testing_accu])

	for b2 in beta2:
		name = "beta1=0, beta2=" + str(b2) + ", epsilon=0"
		training_accu, valid_accu, testing_accu, _, _, _, _, _, _, _ = buildGraph(beta2=b2, lossType=lossType, learning_rate=0.001, optimizerType='ADAM')
		inv_graph.append([name, training_accu, valid_accu, testing_accu])

	for e in epsilon:
		name = "beta1=0, beta2=0, epsilon=" + str(e)
		training_accu, valid_accu, testing_accu, _, _, _, _, _, _, _ = buildGraph(epsilon=e, lossType=lossType, learning_rate=0.001, optimizerType='ADAM')
		inv_graph.append([name, training_accu, valid_accu, testing_accu])


	for i in range(len(inv_graph)):
		print(inv_graph[i][0])
		print('training accuracy: ', inv_graph[i][1])
		print('validation accuracy: ', inv_graph[i][2])
		print('testing accuracy: ', inv_graph[i][3])
		print(" ")

	plt.show()



# # PART 1
# trainingData, validationData, testingData, trainingTarget, validationTarget, testingTarget = loadData()

# weight = np.zeros((trainingData.shape[1] * trainingData.shape[2], 1))
# bias = np.zeros((1,1))
# trainData = trainingData.reshape(trainingData.shape[0], -1)
# testData = testingData.reshape(testingData.shape[0], -1)
# validData = validationData.reshape(validationData.shape[0], -1)

# W, b = grad_descent(weight, bias, trainData, trainingTarget, testData, testingTarget, validData, validationTarget, alpha=0.001, iterations=5000, reg=0.001, EPS=1e-7, lossType="MSE")

# accuracy(weight, bias, trainData, trainingTarget)


# # PART 2
# W, b = grad_descent(weight, bias, trainData, trainingTarget, testData, testingTarget, validData, validationTarget, alpha=0.001, iterations=5000, reg=0.001, EPS=1e-7, lossType="CE")

# accuracy(weight, bias, trainData, trainingTarget)


# # PART 3
# SGD('CE', 'GD')
# batch_size_investigation('CE')
hyperparameter_investigation('CE')


# ref: https://chunml.github.io/ChunML.github.io/tutorial/Regularization/
# ref: https://en.wikipedia.org/wiki/Linear_regression
# ref: https://chunml.github.io/ChunML.github.io/tutorial/Regularization/