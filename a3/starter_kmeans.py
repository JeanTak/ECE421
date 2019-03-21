import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

# Loading data
data = np.load('data2D.npy')
#data = np.load('data100D.npy')
[num_pts, dim] = np.shape(data)

is_valid = True
# For Validation set
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]


# Distance function for K-means
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)

		# ref: https://esciencegroup.com/2016/01/05/an-encounter-with-googles-tensorflow/

    expanded_data = tf.expand_dims(X, 0)
    expanded_centroids = tf.expand_dims(MU, 1)
    distances = tf.reduce_sum(tf.square(tf.subtract(expanded_data, expanded_centroids)), 2)
    return tf.transpose(distances)
    


def graph_plot(data, centroids, distances, k, loss_list):
    
	# LOSS
	plt.title('k-means (k = ' + str(k) + ')')
	plt.grid(True)
	plt.xlabel('epochs')
	plt.ylabel('loss')
	plt.plot(loss_list)
	plt.show()
	
	#DISTRIBUTION
	
	#nearest cluster index
	cluster_index_list = np.argmin(distances, axis = 1)

	#calculate distributions of the clusters in our samples
	cluster_distributions = []
	
	for index in range(k):
		distribution = (cluster_index_list==index).sum() / data.shape[0]
		cluster_distributions.append(distribution)
		print('Cluster '+ str(index) + ': ' + str(distribution * 100) + '% of samples' )

	# Plot Graph	
	plt.title('k-means clustering distribution (k = ' + str(k) + ')')
	plt.grid(True)

	plt.scatter(data[:,0], data[:,1], c=cluster_index_list, linewidths=0, s=8, cmap='Set1')
	plt.scatter(centroids[:,0], centroids[:,1], c=cluster_distributions, s=50, vmin=0, vmax=np.max(cluster_distributions), cmap='Blues')
	plt.colorbar()
	
	#Cluster Tags
	cluster = 0
	for centroid in centroids:
		plt.text(centroid[0], centroid[1]-0.2, 'Cluster '+ str(cluster), color="black", verticalalignment='top', fontsize=15)
		cluster += 1
	plt.show()



def validation_loss(centroids):
	
	tf.reset_default_graph()

	#calculate distances to clusters
	distances = distanceFunc(val_data, centroids)
	distance_to_clusters = tf.reduce_min(distances, axis=1)

	#loss is the total distances
	loss = tf.reduce_sum(distance_to_clusters)
	
	with tf.Session() as session:
		tf.initializers.global_variables().run()
		valid_loss = session.run(loss)
	
	return valid_loss
    


def k_means_training(k, learning_rate=0.1):

	tf.reset_default_graph()
	#Centroids: dimension: K x D and normal distribution
	centroids = tf.Variable(tf.random_normal(np.array([k, dim]), dtype=data.dtype))

	#calculate distances to clusters
	distances = distanceFunc(data[:], centroids)
	cluster_distances = tf.reduce_min(distances, axis=1)

	# Total Distances -> Loss
	loss = tf.reduce_sum(cluster_distances)

	#Optimizer
	optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(loss)
	
	with tf.Session() as session:
		
		tf.initializers.global_variables().run()

		epoch = 0
		diff, prev_loss = float('inf'), float('inf')
		training_loss_list = []

		# THRESHOLD: 1e-7
		while diff > 1e-7:
			
			updated_centroids, current_loss, d, _ = session.run([centroids, loss, distances, optimizer])

			training_loss_list.append(current_loss)
			
			# the change of percentage between current loss and the previous loss
			diff = abs((prev_loss - current_loss) / current_loss)
			
			prev_loss = current_loss
			epoch += 1
				
		print("Total epoches: ", epoch)
		print("Final total training loss: ", prev_loss)
		print("Average loss:", prev_loss / data.shape[0])

	graph_plot(data, updated_centroids, d, k, training_loss_list)

	return updated_centroids


centroids = k_means_training(k=3)

if is_valid:
    valid_loss = validation_loss(centroids)
    print('Total validation loss: ', valid_loss)
    print('Average loss: ', valid_loss / val_data.shape[0])
    