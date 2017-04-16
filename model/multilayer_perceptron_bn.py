import numpy as np
import tensorflow as tf
import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

learning_rate = 0.01
momentum_rate = 0.9

def dense(x, size, scope):
	return tf.contrib.layers.fully_connected(x, size, 
											 activation_fn=None,
											 scope=scope)

def dense_batch_relu(x, phase, scope):
	with tf.variable_scope(scope):
		h1 = tf.contrib.layers.fully_connected(x, 100, 
											   activation_fn=None,
											   scope='dense')
		h2 = tf.contrib.layers.batch_norm(h1, 
										  center=True, scale=True, 
										  is_training=phase,
										  scope='bn')
		return tf.nn.relu(h2, 'relu')

tf.reset_default_graph()
x = tf.placeholder('float32', (None, 784), name='x')
y = tf.placeholder('float32', (None, 10), name='y')
phase = tf.placeholder(tf.bool, name='phase')

h1 = dense_batch_relu(x, phase,'layer1')
h2 = dense_batch_relu(h1, phase, 'layer2')
logits = dense(h2, 10, 'logits')

with tf.name_scope('accuracy'):
	accuracy = tf.reduce_mean(tf.cast(
			tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1)),
			'float32'))

with tf.name_scope('loss'):
	loss = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(logits, y))

optimizers = [
(tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss), "GradientDescentOptimizer"),
(tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss), "AdadeltaOptimizer"),
(tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(loss), "AdagradOptimizer"),
(tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum_rate).minimize(loss), "MomentumOptimizer"),
(tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss), "AdamOptimizer"),
(tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss), "RMSPropOptimizer"),
]

def train(optimizer):
	print optimizer[1]
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		# Ensures that we execute the update_ops before performing the train_step
		# train_step = tf.train.AdamOptimizer(0.01).minimize(loss)
		train_step = optimizer[0]
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	
	log_file = open("./log/"+optimizer[1], "w")
	iterep = 500
	for i in range(iterep * 50):
		x_train, y_train = mnist.train.next_batch(100)
		sess.run(train_step,
				 feed_dict={'x:0': x_train, 
							'y:0': y_train, 
							'phase:0': 1})
		if (i + 1) %  iterep == 0:
			epoch = (i + 1)/iterep
			cost, training_acc = sess.run([loss, accuracy], 
						  feed_dict={'x:0': mnist.train.images,
									 'y:0': mnist.train.labels,
									 'phase:0': 1})
			validation_acc= sess.run(accuracy, 
						 feed_dict={'x:0': mnist.test.images,
									'y:0': mnist.test.labels,
									'phase:0': 0})
			log_file.write("Epoch:"+str('%02d'%(epoch))+", "+\
					"cost="+str("{:.4f}".format(cost))+", "+\
					"training_acc="+str("{:.4}".format(training_acc))+", "+\
					"validation_acc="+str("{:.4}".format(validation_acc))+"\n")
	log_file.close()


if __name__ == "__main__":
	for optimizer in optimizers:
		train(optimizer)
