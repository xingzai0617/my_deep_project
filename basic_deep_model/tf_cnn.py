# -----------------------------------------------------------------------------------------------------
'''
@author:	hongwen sun
&usage:		TensorFlow练习之卷积神经网络
'''
# -----------------------------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist/", one_hot = True, reshape = False)

# =============定义网络结构==============
def conv2d(x, w, b):
	return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME') + b

def pool(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	
# =============定义网络结构==============
def cnn_net(x, keepprob):
	w1 = tf.Variable(tf.random_normal([5, 5, 1, 64]))
	b1 = tf.Variable(tf.random_normal([64]))
	w2 = tf.Variable(tf.random_normal([5, 5, 64, 32]))
	b2 = tf.Variable(tf.random_normal([32]))
	w3 = tf.Variable(tf.random_normal([7*7*32, 10]))
	b3 = tf.Variable(tf.random_normal([10]))
	hidden1 = pool(conv2d(x, w1, b1))
	hidden1 = tf.nn.dropout(hidden1, keepprob)
	hidden2 = pool(conv2d(hidden1, w2, b2))
	hidden2 = tf.reshape(hidden2, [-1, 7*7*32])
	hidden2 = tf.nn.dropout(hidden2, keepprob)
	output = tf.matmul(hidden2, w3) + b3
	return output


# ==========定义所需占位符=============
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y_ = tf.placeholder(tf.float32,[None, 10])
keepprob = tf.placeholder(tf.float32)

# =============学习率优化==============
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.01, global_step, 100, 0.96, staircase=True)

# ============训练所需损失函数==========
logits = cnn_net(x, keepprob)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))
opt = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

# ==============定义评估模型============
predict = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))


batch_size = 100
batch_nums = mnist.train.labels.shape[0] // batch_size
epochs = 10

# ================开始训练==============
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for k in range(epochs):
		for i in range(batch_nums):
			# ==============梯度下降进行训练=============================
			xs, ys = mnist.train.next_batch(batch_size)
			sess.run(opt, {x: xs, y_:ys, keepprob:0.75})
			# ==============评估模型在验证集上的识别率====================
			if i % 50 == 0:
				acc = sess.run(accuracy, {x: mnist.validation.images[:1000], y_:mnist.validation.labels[:1000], keepprob:1.})
				print(k, 'epochs, ', i, 'iters, ', ', acc :', acc)
