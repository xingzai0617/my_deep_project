# -*- coding: UTF-8 -*-
import time
import os
import numpy as np
import tensorflow as tf
import re
from utils import is_uchar

# ==============================数据预处理=============================

# ========读取原始数据========
with open('data.txt', 'r', encoding='utf-8') as f:
	data = f.readlines()

# ==========处理数据==========
# 将.....替换为句号
data = [line.replace('……', '。') for line in data if len(line) > 1]

# 生成一个正则，负责找'()'包含的内容，并将其替换为空
pattern = re.compile(r'\(.*\)')
data = [pattern.sub('', lines) for lines in data]
# 只保留没有乱码的数据
data = ''.join(data)
data = [char for char in data if is_uchar(char)]
data = ''.join(data[:])

with open('newdata.txt', 'w', encoding='utf-8') as f:
	f.write(data)

# =====生成字典=====
vocab = set(data)
int_to_vocab = list(vocab)
char2id = {c:i for i,c in enumerate(vocab)}

with open('dict.txt', 'w', encoding='utf-8') as f:
	f.write('\n'.join(vocab))
# =====转换数据为数字格式======
numdata = [char2id[char] for char in data]
numdata = np.array(numdata)


def get_batches(arr, n_seqs, n_steps):    
    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)
    arr = arr[:batch_size * n_batches]
    # 重塑
    arr = arr.reshape((n_seqs, -1))
    for n in range(0, arr.shape[1], n_steps):
        x = arr[:, n:n+n_steps]
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y

train_data = get_batches(numdata, 4, 5)
print(next(train_data))



'''===========================实现语言模型========================='''

HIDDEN_SIZE = 200
NUM_LAYERS = 2
VOCAB_SIZE = len(vocab)
LEARNING_RATE = 1.
TRAIN_BATCH_SIZE = 16
TRAIN_NUM_STEP = 100

# 测试时不需要截断，可以将测试数据看成一个超长序列
EVAL_BATCH_SIZE = 1
EVAL_NUM_STEP = 1
NUM_EPOCH = 100
KEEP_PROB = 0.8
MAX_GRAD_NORM = 5


# 通过一个类来描述模型，好维护神经网络中的额状态
class PTBModel(object):
	"""docstring for PTBModel"""
	def __init__(self, is_training, batch_size, num_steps):
		super(PTBModel, self).__init__()
		self.is_training = is_training
		self.batch_size = batch_size
		self.num_steps = num_steps
		# 定义输入占位符和训练所需的标签
		self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
		self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

		# 定义rnn的单元
		lstm_cell = tf.contrib.rnn.LSTMCell(HIDDEN_SIZE)
		# 训练过程中需要加入dropout防止过拟合
		if is_training:
			lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=KEEP_PROB)
		# 利用multirnn构建多层的rnn结构
		cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * NUM_LAYERS)

		# 初始化模型的状态为零状态
		self.initial_state = cell.zero_state(batch_size, tf.float32)
		# 定义embedding层
		embedding = tf.get_variable('embedding', [VOCAB_SIZE, HIDDEN_SIZE])
		# 将输入通过embed lookup获得词向量
		inputs = tf.nn.embedding_lookup(embedding, self.input_data)
		# 如果训练模式下也需要将输入进行dropout
		if is_training:
			inputs = tf.nn.dropout(inputs, KEEP_PROB)
		# 定义输出
		outputs = []
		state = self.initial_state
		with tf.variable_scope('RNN'):
			for time_step in range(num_steps):
				if time_step > 0: tf.get_variable_scope().reuse_variables()
				# 获得rnn结构的输出，包括输出和状态
				cell_output, state = cell(inputs[:, time_step, :], state)
				# 获得整个序列的输出
				outputs.append(cell_output)
				
		output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])
		# 构建全连接层，将output映射到具体的的单词上去
		weight = tf.get_variable('weight', [HIDDEN_SIZE, VOCAB_SIZE])
		bias = tf.get_variable('bias', [VOCAB_SIZE])
		# 最终的输出结果
		logits = tf.matmul(output, weight) + bias
		# 利用sequence_loss_by_example计算序列的损失值之和
		loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [tf.reshape(self.targets, [batch_size * num_steps])], 
													[tf.ones([batch_size * num_steps], dtype=tf.float32)])
		# 平均的cost
		self.logits = logits
		self.cost = tf.reduce_sum(loss) / batch_size
		self.final_state = state
		# 训练所需模块
		if not is_training: return
		trainable_variables = tf.trainable_variables()
		# 限制梯度，防止梯度爆炸
		grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM)
		# 定义优化器 
		# AdagradOptimizer MomentumOptimizer RMSPropOptimizer AdamOptimizer AdadeltaOptimizer
		optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
		# 应用梯度下降
		self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))


'''===========================训练模型========================='''

def run_epoch(session, model, data, train_op, output_log):
	total_costs = 0.
	iters = 0
	state = session.run(model.initial_state)
	for step, (x, y) in enumerate(get_batches(data, model.batch_size, model.num_steps)):
		predict, cost, state, _ = session.run([model.logits, model.cost, model.final_state, train_op], 
									{model.input_data: x, model.targets: y, model.initial_state: state})
		total_costs += cost
		iters += model.num_steps
		if output_log and step % 50 == 0:
			print('after ', step, ' steps, perplexity is ', np.exp(total_costs / iters))
			print(''.join([int_to_vocab[c] for c in np.argmax(predict, axis=1)]))
	return np.exp(total_costs / iters)



'''===========================主函数========================='''

def main():
	train_data = numdata
	initializer = tf.random_uniform_initializer(-0.05, 0.05)
	with tf.variable_scope('language_model', reuse=None, initializer=initializer):
		train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)
	with tf.variable_scope('language_model', reuse=True, initializer=initializer):
		eval_model = PTBModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)
	saver = tf.train.Saver()
	with tf.Session() as sess:
		# 初始化变量
		sess.run(tf.global_variables_initializer())
		for i in range(NUM_EPOCH):
			print('in iteration:', (i+1))
			run_epoch(sess, train_model, train_data, train_model.train_op, True)
			saver.save(sess, 'model/model.cpk')


if __name__ == '__main__':
	main()


