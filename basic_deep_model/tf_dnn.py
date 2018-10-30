# -----------------------------------------------------------------------------------------------------
'''
@author:	hongwen sun
&usage:		TensorFlow练习之全连接神经网络
'''
# -----------------------------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("mnist/", one_hot = True)

print('train image shape:', mnist.train.images.shape, 'trian label shape:', mnist.train.labels.shape)
print('val image shape:', mnist.validation.images.shape)
print('test image shape:', mnist.test.images.shape)

input_size = 784
hidden1_size = 512
hidden2_size = 256
output_size = 10
learning_rate_base = 0.005
epochs = 2
batch_size = 1000
batch_nums = mnist.train.labels.shape[0] // batch_size

# =============定义网络结构==============
def dense(x, w, b, keepprob, name):
	return tf.nn.dropout(tf.nn.relu(tf.matmul(x, w) + b, name=name), keepprob)


def DNNModel(images, w, b, keepprob):
	with tf.name_scope('dense1'):
		dense1 = dense(images, w[0], b[0], keepprob, name='dense1')
	with tf.name_scope('dense2'):
		dense2 = dense(dense1, w[1], b[1], keepprob, name='dense2')
	with tf.name_scope('output_layer'):
		output = tf.matmul(dense2, w[2]) + b[2]
	return output

def gen_weights(units_list):
	w = []
	b = []
	for i in range(len(units_list)-1):
		sub_w = tf.Variable(tf.random_normal([units_list[i], units_list[i+1]]), name='weight'+str(i))
		sub_b = tf.Variable(tf.random_normal([units_list[i+1]]), name='bias'+str(i))
		w.append(sub_w)
		b.append(sub_b)
	return w, b

# ==========定义所需占位符=============
with tf.name_scope('input'):
	x = tf.placeholder(tf.float32, [None, 784], name='input')
	keepprob = tf.placeholder(tf.float32, name='keep_prob')
	y_ = tf.placeholder(tf.float32, [None, 10], name='labels')

# =============定义变量===============
units_list = [784, 512, 256, 10]
with tf.name_scope('dense_variables'):
	w, b = gen_weights(units_list)

# =============学习率优化==============
with tf.name_scope('learning_rateopt'):
	global_step = tf.Variable(0)
	learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, 10, 0.96, staircase=True)

# ============训练所需损失函数==========
logits = DNNModel(x, w, b, keepprob)
with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))
	tf.summary.scalar('loss', loss)
with tf.name_scope('AdamOptimizer'):
	opt = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

# ==============定义评估模型============
with tf.name_scope('accuracy'):
	predict = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
	acc = tf.reduce_mean(tf.cast(predict, tf.float32))
	tf.summary.scalar('accuracy', acc)

merged = tf.summary.merge_all()
# ================开始训练==============
saver = tf.train.Saver()
with tf.Session() as sess:
	# ==============tensorboard============
	writer = tf.summary.FileWriter('logs/tensorboard', tf.get_default_graph())

	sess.run(tf.global_variables_initializer())
	for i in range(batch_nums * epochs):
		# 训练模型，输入为一个batch的数据
		xs, ys = mnist.train.next_batch(batch_size)
		summary, _ = sess.run([merged, opt], {x: xs, y_: ys, keepprob: 0.75})
		writer.add_summary(summary, i)
		# 评估模型在验证集上的识别率
		if i % 50 == 0:
			# 配置tensorboard的节点信息
			run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
			run_metadata = tf.RunMetadata()
			sess.run(opt, {x: xs, y_: ys, keepprob: 0.75}, options=run_options, run_metadata=run_metadata)
			# 验证集
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

writer.close()