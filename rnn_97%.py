import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
#from tensorflow.python.ops import rnn,rnn_cell
#from tensorflow.python.ops import rnn
from tensorflow.contrib import rnn
#from tensorflow.contrib.rnn.python.ops import rnn,rnn_cell
#from tensorflow.contrib.rnn import rnn_cell

mnist = input_data.read_data_sets("data/mnist", one_hot=True)



no_of_epochs = 10
no_of_classes = 10
batch_size = 128
chunk_size = 28
no_of_chunks = 28
rnn_size = 128

x = tf.placeholder('float',[None,no_of_chunks,chunk_size])
y = tf.placeholder('float')

def recurrent_neural_network(x):
	layer = {'weights':tf.Variable(tf.random_normal([rnn_size,no_of_classes])),
					   'biases':tf.Variable(tf.random_normal([no_of_classes]))}
	x = tf.transpose(x,[1,0,2])
	x = tf.reshape(x,[-1,chunk_size])
	x = tf.split(x,no_of_chunks,0)
	#commented lines works for older versions
	#lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True) 
	lstm_cell = rnn.BasicLSTMCell(rnn_size)
	outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
	
	#lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
	#outputs,states = rnn.rnn_cell(lstm_cell,x,dtype = tf.float32)	

	output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']
	
	return output
def train_neural_network(x):
	prediction = recurrent_neural_network(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = y) )
	learning_rate = 0.1
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(no_of_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x,epoch_y = mnist.train.next_batch(batch_size)
				epoch_x = epoch_x.reshape((batch_size,no_of_chunks,chunk_size))
				_, c = sess.run([optimizer,cost],feed_dict = {x:epoch_x, y:epoch_y})
				epoch_loss += c
			print('Epoch',epoch,'completed out of',no_of_epochs,'loss:',epoch_loss)
		correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print('Accuracy:',accuracy.eval({x:mnist.test.images.reshape((-1,no_of_chunks,chunk_size)) ,y:mnist.test.labels}))

train_neural_network(x)




