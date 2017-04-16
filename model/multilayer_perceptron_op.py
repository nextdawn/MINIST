'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

# Parameters
learning_rate0 = 0.001
learning_rate1 = 0.01
learning_rate2 = 0.1
momentum_rate = 0.9
training_epochs = 50
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
	# Hidden layer with RELU activation
	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	layer_1 = tf.nn.relu(layer_1)
	# Hidden layer with RELU activation
	layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
	layer_2 = tf.nn.relu(layer_2)
	# Output layer with linear activation
	out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
	return out_layer

# Store layers weight & bias
weights = {
	'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
	'b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'b2': tf.Variable(tf.random_normal([n_hidden_2])),
	'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), "float"))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
optimizers = [
(tf.train.GradientDescentOptimizer(learning_rate=learning_rate0).minimize(cost), "GradientDescentOptimizer_"+str(learning_rate0)),
(tf.train.AdadeltaOptimizer(learning_rate=learning_rate0).minimize(cost), "AdadeltaOptimizer_"+str(learning_rate0)),
(tf.train.AdagradOptimizer(learning_rate=learning_rate0).minimize(cost), "AdagradOptimizer_"+str(learning_rate0)),
(tf.train.MomentumOptimizer(learning_rate=learning_rate0, momentum=momentum_rate).minimize(cost), "MomentumOptimizer_"+str(learning_rate0)),
(tf.train.AdamOptimizer(learning_rate=learning_rate0).minimize(cost), "AdamOptimizer_"+str(learning_rate0)),
(tf.train.RMSPropOptimizer(learning_rate=learning_rate0).minimize(cost), "RMSPropOptimizer_"+str(learning_rate0)),

(tf.train.GradientDescentOptimizer(learning_rate=learning_rate1).minimize(cost), "GradientDescentOptimizer_"+str(learning_rate1)),
(tf.train.AdadeltaOptimizer(learning_rate=learning_rate1).minimize(cost), "AdadeltaOptimizer_"+str(learning_rate1)),
(tf.train.AdagradOptimizer(learning_rate=learning_rate1).minimize(cost), "AdagradOptimizer_"+str(learning_rate1)),
(tf.train.MomentumOptimizer(learning_rate=learning_rate1, momentum=momentum_rate).minimize(cost), "MomentumOptimizer_"+str(learning_rate1)),
(tf.train.AdamOptimizer(learning_rate=learning_rate1).minimize(cost), "AdamOptimizer_"+str(learning_rate1)),
(tf.train.RMSPropOptimizer(learning_rate=learning_rate1).minimize(cost), "RMSPropOptimizer_"+str(learning_rate1)),

(tf.train.GradientDescentOptimizer(learning_rate=learning_rate2).minimize(cost), "GradientDescentOptimizer_"+str(learning_rate2)),
(tf.train.AdadeltaOptimizer(learning_rate=learning_rate2).minimize(cost), "AdadeltaOptimizer_"+str(learning_rate2)),
(tf.train.AdagradOptimizer(learning_rate=learning_rate2).minimize(cost), "AdagradOptimizer_"+str(learning_rate2)),
(tf.train.MomentumOptimizer(learning_rate=learning_rate2, momentum=momentum_rate).minimize(cost), "MomentumOptimizer_"+str(learning_rate2)),
(tf.train.AdamOptimizer(learning_rate=learning_rate2).minimize(cost), "AdamOptimizer_"+str(learning_rate2)),
(tf.train.RMSPropOptimizer(learning_rate=learning_rate2).minimize(cost), "RMSPropOptimizer_"+str(learning_rate2)),
]

# Initializing the variables
init = tf.global_variables_initializer()

for optimizer in optimizers:
	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)
	
		def test_model():
			# Test model
			correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
			# Calculate accuracy
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
			return  accuracy.eval({x: mnist.test.images, y: mnist.test.labels})

		log_file = open("./log/"+optimizer[1], "w")
		# Training cycle
		for epoch in range(training_epochs):
			print(optimizer[1], ":", epoch)
			avg_cost = 0.
			avg_acc = 0.
			total_batch = int(mnist.train.num_examples/batch_size)
			# Loop over all batches
			for i in range(total_batch):
				batch_x, batch_y = mnist.train.next_batch(batch_size)
				# Run optimization op (backprop) and cost op (to get loss value)
				_, c, a = sess.run([optimizer[0], cost, acc], feed_dict={x: batch_x,
															  y: batch_y})
				# Compute average loss
				avg_cost += c / total_batch
				avg_acc += a / total_batch
			# Display logs per epoch step
			if epoch % display_step == 0:
				validation_acc = test_model()
				log_file.write("Epoch:"+str('%02d'%(epoch+1))+", "+\
					"cost="+str("{:.4f}".format(avg_cost))+", "+\
					"training_acc="+str("{:.4}".format(avg_acc))+", "+\
					"validation_acc="+str("{:.4}".format(validation_acc))+"\n")
		log_file.close()
