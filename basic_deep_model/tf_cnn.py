# -----------------------------------------------------------------------------------------------------
'''
@author:	hongwen sun
&usage:		TensorFlow练习之卷积神经网络
'''
# -----------------------------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist/", one_hot = True, reshape = False)

#input tensor of shape [batch, in_height, in_width, in_channels]
#filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
def conv2d(image, w, b, name):
	return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(image, w, strides=[1,1,1,1], padding='SAME'), b), name=name)

#Tensor of shape [batch, height, width, channels] 
def pooling(featmamps, kernel_size, name):
	return tf.nn.max_pool(featmamps, [1, kernel_size, kernel_size, 1], [1, kernel_size, kernel_size, 1], padding='SAME')

def batch_norm(featmamps, is_train):
	return tf.layers.batch_normalization(featmamps, training=is_train)

w_alpha=0.01
b_alpha=0.1

weights = {
	'wc1' : tf.get_variable(name='wc1', shape=[3, 3, 1, 32], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
	'wc2' : tf.get_variable(name='wc2', shape=[3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
	'wc3' : tf.get_variable(name='wc3', shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
	'wc4' : tf.get_variable(name='wc4', shape=[3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
	#全连接参数
	'wd1' : w_alpha*tf.Variable(tf.random_normal([2*2*128, 256])),
	'wd2' : w_alpha*tf.Variable(tf.random_normal([256, 100])),
	'out' : w_alpha*tf.Variable(tf.random_normal([100, 10]))
}

biases = {
	'bc1' : b_alpha*tf.Variable(tf.random_normal([32])),
	'bc2' : b_alpha*tf.Variable(tf.random_normal([64])),
	'bc3' : b_alpha*tf.Variable(tf.random_normal([64])),
	'bc4' : b_alpha*tf.Variable(tf.random_normal([128])),
	'bd1' : b_alpha*tf.Variable(tf.random_normal([256])),
	'bd2' : b_alpha*tf.Variable(tf.random_normal([100])),
	'out' : b_alpha*tf.Variable(tf.random_normal([10])),
}

def constructNet(images,weights,biases,is_training):
	#reshape image
	images = tf.reshape(images, [-1,28,28,1])
	#1st cnn layer
	conv1 = conv2d(images, weights['wc1'], biases['bc1'], 'conv1')
	conv1 = batch_norm(conv1, is_training)
	conv1 = tf.nn.relu(conv1)
	conv1 = pooling(conv1, 2, 'pool1')
	#2nd cnn layer
	conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], 'conv2')
	conv2 = batch_norm(conv2, is_training)
	conv2 = tf.nn.relu(conv2)
	conv2 = pooling(conv2, 2, 'pool2')
	#3rd cnn layer
	conv3 = conv2d(conv2, weights['wc3'], biases['bc3'], 'conv3')
	conv3 = batch_norm(conv3, is_training)
	conv3 = tf.nn.relu(conv3)
	conv3 = pooling(conv3, 2, 'pool3')
	#4th cnn layer
	conv4 = conv2d(conv3, weights['wc4'], biases['bc4'], 'conv4')
	conv4 = batch_norm(conv4, is_training)
	conv4 = tf.nn.relu(conv4)
	conv4 = pooling(conv4, 2, 'pool4')
	#全连接1
	dense1 = tf.reshape(conv4, [-1, weights['wd1'].get_shape().as_list()[0]])
	dense1 = tf.matmul(dense1, weights['wd1']) + biases['bd1']
	dense1 = batch_norm(dense1, is_training)
	dense1 = tf.nn.relu(dense1)
	dense1 = tf.nn.dropout(dense1, 0.8, noise_shape=None, seed=None, name=None)
	#全连接2
	dense2 = tf.nn.relu(tf.matmul(dense1, weights['wd2']) + biases['bd2'])
	dense2 = tf.nn.dropout(dense2, 0.8, noise_shape=None, seed=None, name=None)
	#全连接3
	out = tf.matmul(dense2, weights['out']) + biases['out']
	return out


def train(num_batches, batch_size, learning_rate):
	# model
	logits = constructNet(x, weights, biases, is_training)
	# loss function: cross entropy, ...
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
	opts = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(opts):
		train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

	# 定义评估模型
	correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# 开始训练
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for batch_i in range(num_batches):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			# train this batch
			sess.run(train_opt, {x:batch_xs, y:batch_ys, is_training:True})
			if batch_i % 100 == 0:
				gloss, acc = sess.run([loss, accuracy], {x:mnist.validation.images, y:mnist.validation.labels, is_training:False})
				print('batch:{:>2}: validation loss:{:>3.5f}, validation accuracy:{:>3.5f}'.format(batch_i, gloss, acc))
			elif batch_i % 25 == 0:
				gloss, acc = sess.run([loss, accuracy], {x:batch_xs, y:batch_ys, is_training:False})
				print('batch:{:>2}: train loss:{:>3.5f}, train accuracy:{:>3.5f}'.format(batch_i, gloss, acc))



x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])
is_training = tf.placeholder(tf.bool)

num_batches = 800
batch_size = 64
learning_rate = 0.001

train(num_batches, batch_size, learning_rate)