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

def neural_network_model(data):
	hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([784,no_of_nodes_L1])),
					   'biases':tf.Variable(tf.random_normal([no_of_nodes_L1]))}
	hidden_layer_2 = {'weights':tf.Variable(tf.random_normal([no_of_nodes_L1,no_of_nodes_L2])),
					   'biases':tf.Variable(tf.random_normal([no_of_nodes_L2]))}
	hidden_layer_3 = {'weights':tf.Variable(tf.random_normal([no_of_nodes_L2,no_of_nodes_L3])),
					   'biases':tf.Variable(tf.random_normal([no_of_nodes_L3]))}
	output_layer = {'weights':tf.Variable(tf.random_normal([no_of_nodes_L3,no_of_classes])),
					   'biases':tf.Variable(tf.random_normal([no_of_classes]))}

	layer_1 = tf.add(tf.matmul(data,hidden_layer_1['weights']),hidden_layer_1['biases'])
	layer_1 = tf.nn.relu(layer_1)

	layer_2 = tf.add(tf.matmul(layer_1,hidden_layer_2['weights']),hidden_layer_2['biases'])
	layer_2 = tf.nn.relu(layer_2)

	layer_3 = tf.add(tf.matmul(layer_2,hidden_layer_3['weights']),hidden_layer_3['biases'])
	layer_3 = tf.nn.relu(layer_3)

	output = tf.matmul(layer_3,output_layer['weights']) + output_layer['biases']
	
	return output
def train_neural_network(x):
	prediction = neural_network_model(x)
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
