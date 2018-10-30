# -----------------------------------------------------------------------------------------------------
'''
@author:	hongwen sun
&usage:		TensorFlow练习之循环神经网络
'''
# -----------------------------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("mnist/", one_hot = True)

print('train image shape:', mnist.train.images.shape, 'trian label shape:', mnist.train.labels.shape)
print('val image shape:', mnist.validation.images.shape)
print('test image shape:', mnist.test.images.shape)

epochs = 10
batch_size = 1000
batch_nums = mnist.train.labels.shape[0] // batch_size

# =============定义网络结构==============
def rnn(x, batch_size, keepprob):
	hidden_size = 28
	rnn_layers = 2
	rnn_cell = tf.contrib.rnn.LSTMCell(28)
	rnn_drop = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=keepprob)
	multi_cell = tf.contrib.rnn.MultiRNNCell([rnn_drop] * 2)
	state = multi_cell.zero_state(batch_size, tf.float32)
	outputs, states = tf.nn.dynamic_rnn(multi_cell, x, initial_state=state)
	w = tf.Variable(tf.random_normal([28, 10]))
	b = tf.Variable(tf.random_normal([10]))
	output = tf.matmul(outputs[:,-1,:], w) + b
	return output, states

# ==========定义所需占位符=============
x = tf.placeholder(tf.float32, [None, 28, 28])
y_ = tf.placeholder(tf.float32, [None, 10])
keepprob = tf.placeholder(tf.float32)

# =============学习率优化==============
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.01, global_step, 10, 0.96, staircase=True)

# ============训练所需损失函数==========
logits, states = rnn(x, batch_size, keepprob)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))
opt = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

# ==============定义评估模型============
predict = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
acc = tf.reduce_mean(tf.cast(predict, tf.float32))


# ================开始训练==============
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for k in range(epochs):
		for i in range(batch_nums):
			# ==============梯度下降进行训练=============================
			xs, ys = mnist.train.next_batch(batch_size)
			sess.run(opt, {x: xs.reshape((-1, 28, 28)), y_: ys, keepprob: 0.8})
			# ==============评估模型在验证集上的识别率====================
			if i % 50 == 0:
				val_losses = 0
				accuracy = 0
				xv, yv = mnist.validation.next_batch(batch_size)
				for i in range(xv.shape[0]):
					val_loss, accy = sess.run([loss, acc], {x: xv.reshape((-1, 28, 28)), y_: yv, keepprob: 1.})
					val_losses += val_loss
					accuracy += accy
				print('val_loss is :', val_losses / xv.shape[0], ', accuracy is :', accuracy / xv.shape[0])
