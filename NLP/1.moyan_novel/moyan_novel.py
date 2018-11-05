
# ===============================读取原始数据=============================
with open('data.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()
print(data[0])


# =================================数据清理===============================
import re
# 生成一个正则，负责找'()'包含的内容
pattern = re.compile(r'\(.*\)')
# 将其替换为空
data = [pattern.sub('', lines) for lines in data]
print(data[0])

# 将.....替换为句号
data = [line.replace('……', '。') for line in data if len(line) > 1]
print(data[0])

# ==============判断char是否是乱码===================
def is_uchar(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
            return True
    """判断一个unicode是否是数字"""
    if uchar >= u'\u0030' and uchar<=u'\u0039':
            return True       
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u007a'):
            return True
    if uchar in ('，','。','：','？','“','”','！','；','、','《','》','——'):
            return True
    return False


# 将每行的list合成一个长字符串
data = ''.join(data)
data = [char for char in data if is_uchar(char)]
data = ''.join(data)
print(data[:100])


# ======================生成字典=====================
import os
vocab = set(data)
if os.path.exists('vocab.txt'):
	vocab = open('vocab.txt', 'r')
	vocab = vocab.read()

id2char = list(vocab)
char2id = {c:i for i,c in enumerate(vocab)}

print('字典长度:', len(vocab))


import numpy as np
# =======================转换数据为数字格式====================
numdata = [char2id[char] for char in data]
numdata = np.array(numdata)

print('数字数据信息：\n', numdata[:100])
print('\n文本数据信息：\n', ''.join([id2char[i] for i in numdata[:100]]))

# ==========================设计数据生成器=====================
def data_generator(data, batch_size, time_stpes):
	samples_per_batch = batch_size * time_stpes
	batch_nums = len(data) // samples_per_batch
	data = data[:batch_nums*samples_per_batch]
	data = data.reshape((batch_size, batch_nums, time_stpes))
	for i in range(batch_nums):
		x = data[:, i, :]
		y = np.zeros_like(x)
		y[:, :-1] = x[:, 1:]
		try:
			y[:, -1] = data[:, i+1, 0]
		except:
			y[:, -1] = data[:, 0, 0]
		yield x, y

# 打印输出数据
data_batch = data_generator(numdata, 2, 5)
x, y = next(data_batch)
print('input data:', x[0], '\noutput data:', y[0])


import tensorflow as tf
# ====================================搭建模型===================================
class RNNModel():
	"""docstring for RNNModel"""
	def __init__(self, BATCH_SIZE, HIDDEN_SIZE, HIDDEN_LAYERS, VOCAB_SIZE, learning_rate):
		super(RNNModel, self).__init__()
		self.BATCH_SIZE = BATCH_SIZE
		self.HIDDEN_SIZE = HIDDEN_SIZE
		self.HIDDEN_LAYERS = HIDDEN_LAYERS
		self.VOCAB_SIZE = VOCAB_SIZE
		
		# ======定义占位符======
		with tf.name_scope('input'):
			self.inputs = tf.placeholder(tf.int32, [BATCH_SIZE, None])
			self.targets = tf.placeholder(tf.int32, [BATCH_SIZE, None])
			self.keepprb = tf.placeholder(tf.float32)

		# ======定义词嵌入层======
		with tf.name_scope('embedding'):
			embedding = tf.get_variable('embedding', [VOCAB_SIZE, HIDDEN_SIZE])
			emb_input = tf.nn.embedding_lookup(embedding, self.inputs)
			emb_input = tf.nn.dropout(emb_input, self.keepprb)

		# ======搭建lstm结构=====
		with tf.name_scope('rnn'):
			lstm = tf.contrib.rnn.LSTMCell(HIDDEN_SIZE, state_is_tuple=True)
			lstm = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self.keepprb)
			cell = tf.contrib.rnn.MultiRNNCell([lstm] * HIDDEN_LAYERS)
			self.initial_state = cell.zero_state(BATCH_SIZE, tf.float32)
			outputs, self.final_state = tf.nn.dynamic_rnn(cell, emb_input, initial_state=self.initial_state)
            
		# =====重新reshape输出=====
		with tf.name_scope('output_layer'):
			outputs = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])
			w = tf.get_variable('outputs_weight', [HIDDEN_SIZE, VOCAB_SIZE])
			b = tf.get_variable('outputs_bias', [VOCAB_SIZE])
			logits = tf.matmul(outputs, w) + b

		# ======计算损失=======
		with tf.name_scope('loss'):
			self.loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [tf.reshape(self.targets, [-1])], 
															[tf.ones([BATCH_SIZE * TIME_STEPS], dtype=tf.float32)])
			self.cost = tf.reduce_sum(self.loss) / BATCH_SIZE

		# =============优化算法==============
		with tf.name_scope('opt'):
            # =============学习率衰减==============
			global_step = tf.Variable(0)
			learning_rate = tf.train.exponential_decay(learning_rate, global_step, BATCH_NUMS, 0.99, staircase=True)

			# =======通过clip_by_global_norm()控制梯度大小======
			trainable_variables = tf.trainable_variables()
			grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM)
			self.opt = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(grads, trainable_variables))

		# ==============预测输出=============
		with tf.name_scope('predict'):
			self.predict = tf.argmax(logits, 1)


# =======预定义模型参数========
VOCAB_SIZE = len(vocab)
EPOCHS = 50
BATCH_SIZE = 8
TIME_STEPS = 100
BATCH_NUMS = len(numdata) // (BATCH_SIZE * TIME_STEPS)
HIDDEN_SIZE = 512
HIDDEN_LAYERS = 3
MAX_GRAD_NORM = 1
learning_rate = 0.003


# ===========模型训练===========
model = RNNModel(BATCH_SIZE, HIDDEN_SIZE, HIDDEN_LAYERS, VOCAB_SIZE, learning_rate)

# 保存模型
saver = tf.train.Saver()
with tf.Session() as sess:
	writer = tf.summary.FileWriter('logs/tensorboard', tf.get_default_graph())

	sess.run(tf.global_variables_initializer())
	for k in range(EPOCHS):
		state = sess.run(model.initial_state)
		train_data = data_generator(numdata, BATCH_SIZE, TIME_STEPS)
		total_loss = 0.
		for i in range(BATCH_NUMS):
			xs, ys = next(train_data)
			feed = {model.inputs: xs, model.targets: ys, model.keepprb: 0.8, model.initial_state: state}
			costs, state, _ = sess.run([model.cost, model.final_state, model.opt], feed_dict=feed)
			total_loss += costs
			if (i+1) % 50 == 0:
				print('epochs:', k + 1, 'iter:', i + 1, 'cost:', total_loss / i + 1)

	saver.save(sess, './checkpoints/lstm.ckpt')

writer.close()


# ============模型测试============
tf.reset_default_graph()
evalmodel = RNNModel(1, HIDDEN_SIZE, HIDDEN_LAYERS, VOCAB_SIZE, learning_rate)
# 加载模型
saver = tf.train.Saver()
with tf.Session() as sess:
	saver.restore(sess, './checkpoints/lstm.ckpt')
	new_state = sess.run(evalmodel.initial_state)
	x = np.zeros((1, 1)) + 8
	samples = []
	for i in range(100):
		feed = {evalmodel.inputs: x, evalmodel.keepprb: 1., evalmodel.initial_state: new_state}
		c, new_state = sess.run([evalmodel.predict, evalmodel.final_state], feed_dict=feed)
		x[0][0] = c[0]
		samples.append(c[0])
	print('test:', ''.join([id2char[index] for index in samples]))

