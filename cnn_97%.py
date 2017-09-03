import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data/mnist", one_hot=True)

no_of_nodes_L1 = 500
no_of_nodes_L2 = 500
no_of_nodes_L3 = 500

no_of_classes = 10
batch_size = 100

x = tf.placeholder('float',[None,784])
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)
def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def maxpool2d(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def convolution_neural_network(x):
	weights = {'weight_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
			   'weight_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
			   'W_dens_connected':tf.Variable(tf.random_normal([7*7*64,1024])),
			   'out_layer':tf.Variable(tf.random_normal([1024,no_of_classes]))}

	biases = {'biase_conv1':tf.Variable(tf.random_normal([32])),
			   'biase_conv2':tf.Variable(tf.random_normal([64])),
			   'b_dens_connected':tf.Variable(tf.random_normal([1024])),
			   'out_layer':tf.Variable(tf.random_normal([no_of_classes]))}

	x = tf.reshape(x, shape=[-1, 28, 28, 1])
	conv1 = tf.nn.relu(conv2d(x, weights['weight_conv1'])+ biases['biase_conv1'])
	conv1 = maxpool2d(conv1)

	conv2 = tf.nn.relu(conv2d(conv1,weights['weight_conv2'])+ biases['biase_conv2'] )
	conv2 = maxpool2d(conv2)
	dens_connected = tf.reshape(conv2, [-1,7*7*64])
	dens_connected = tf.nn.relu(tf.matmul(dens_connected,weights['W_dens_connected']) + biases['b_dens_connected'])

	dens_connected = tf.nn.dropout(dens_connected,keep_rate)
	output = tf.matmul(dens_connected,weights['out_layer']) + biases['out_layer']
 	
	

	
	return output
def train_neural_network(x):
	prediction = convolution_neural_network(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = y) )
	learning_rate = 0.1
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	no_of_epochs = 10
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(no_of_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x,epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer,cost],feed_dict = {x:epoch_x, y:epoch_y})
				epoch_loss += c
			print('Epoch',epoch,'completed out of',no_of_epochs,'loss:',epoch_loss)
		correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print('Accuracy:',accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))

train_neural_network(x)




