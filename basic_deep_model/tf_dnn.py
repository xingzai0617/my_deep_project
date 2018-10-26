# -----------------------------------------------------------------------------------------------------
'''
@author:	hongwen sun
&usage:		TensorFlow练习之卷积神经网络
'''
# -----------------------------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("mnist/", one_hot = True)

print('train image shape:', mnist.train.images.shape, 'trian label shape:', mnist.train.labels.shape)
print('val image shape:', mnist.validation.images.shape)
print('test image shape:', mnist.test.images.shape)


# -----------------------------------------------------------------------------------------------------
'''
&usage:		DNN网络建模
'''
# -----------------------------------------------------------------------------------------------------

def dense(x, w, b, keepprob, name):
	return tf.nn.dropout(tf.nn.relu(tf.matmul(x, w) + b, name=name), keepprob)


def DNNModel(images, w, b, keepprob):
	dense1 = dense(images, w[0], b[0], keepprob, name='dense1')
	dense2 = dense(dense1, w[1], b[1], keepprob, name='dense2')
	output = tf.matmul(dense2, w[2]) + b[2]
	return output

# -----------------------------------------------------------------------------------------------------
'''
&usage:		定义参数和变量
'''
# -----------------------------------------------------------------------------------------------------
input_size = 784
hidden1_size = 512
hidden2_size = 256
output_size = 10
learning_rate_base = 0.005
epochs = 2
batch_size = 1000
batch_nums = mnist.train.labels.shape[0] // batch_size


x = tf.placeholder(tf.float32, [None, 784])
keepprob = tf.placeholder(tf.float32)
y_ = tf.placeholder(tf.float32, [None, 10])

w1 = tf.Variable(tf.random_normal([input_size, hidden1_size]))
b1 = tf.Variable(tf.random_normal([hidden1_size]))
w2 = tf.Variable(tf.random_normal([hidden1_size, hidden2_size]))
b2 = tf.Variable(tf.random_normal([hidden2_size]))
w_out = tf.Variable(tf.random_normal([hidden2_size, output_size]))
b_out = tf.Variable(tf.random_normal([output_size]))

w = [w1, w2, w_out]
b = [b1, b2, b_out]

# =============学习率优化==============
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, 10, 0.96, staircase=True)

# ============训练所需损失函数==========
logits = DNNModel(x, w, b, keepprob)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))
opt = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

# ==============定义评估模型============
predict = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
acc = tf.reduce_mean(tf.cast(predict, tf.float32))

# -----------------------------------------------------------------------------------------------------
'''
&usage:		开始训练
'''
# -----------------------------------------------------------------------------------------------------
saver = tf.train.Saver()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(batch_nums * epochs):
		# 训练模型，输入为一个batch的数据
		xs, ys = mnist.train.next_batch(batch_size)
		sess.run(opt, {x: xs, y_: ys, keepprob: 0.75})
		# 评估模型在验证集上的识别率
		if i % 50 == 0:
			feeddict = {x: mnist.validation.images, y_: mnist.validation.labels, keepprob: 1.}
			valloss, accuracy = sess.run([loss, acc], feed_dict=feeddict)
			print(i, 'th batch val loss:', valloss, ', accuracy:', accuracy)
	print('test accuracy:', sess.run(acc, {x:mnist.test.images,y_:mnist.test.labels, keepprob:1.}))
	# 保存模型
	saver.save(sess, './checkpoints/tfdnn.ckpt')
	# 导入模型
	saver.restore(sess, './checkpoints/tfdnn.ckpt')
	# 测试测试集上的准确率
	print('test acc:', sess.run(acc, {x:mnist.test.images,y_:mnist.test.labels, keepprob:1.}))