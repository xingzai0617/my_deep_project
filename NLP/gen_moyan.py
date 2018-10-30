# -*- coding: UTF-8 -*-
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
# 只保留没有乱码的数据，包括汉字英文拼音和中文字符
data = ''.join(data)
data = [char for char in data if is_uchar(char)]
data = ''.join(data)

with open('newdata.txt', 'w', encoding='utf-8') as f:
	f.write(data)

# =====生成字典=====
vocab = set(data)
id2char = list(vocab)
char2id = {c:i for i,c in enumerate(vocab)}


# =====转换数据为数字格式======
numdata = [char2id[char] for char in data]
numdata = np.array(numdata)
print(numdata.shape)

# 设计数据生成器
def data_generator(data, batch_size, time_stpes):
	samples_per_batch = batch_size * time_stpes
	batch_nums = len(data) // samples_per_batch
	data = data[:batch_nums*samples_per_batch]
	data = data.reshape((batch_size, batch_nums, time_stpes))
	for i in range(batch_nums):
		x = data[:, i, :]
		y = np.zeros_like(x)
		y[:, :-1] = x[:, 1:]
		y[:, -1] = data[:, i+1, 0]
		yield x, y

# 测试数据生成器
train_data = data_generator(numdata, 4, 5)
print(next(train_data))
print(next(train_data))

# =======预定义模型参数========
BATCH_SIZE = 8
TIME_STEPS = 100
HIDDEN_SIZE = 64
HIDDEN_LAYERS = 2
VOCAB_SIZE = len(vocab)


# ====================================搭建模型===================================

# ======定义占位符======
inputs = tf.placeholder(tf.int32, [None, None])
targets = tf.placeholder(tf.int32, [None, None])
keepprb = tf.placeholder(tf.float32)



# ======定义词嵌入层======
embedding = tf.get_variable('embedding', [VOCAB_SIZE, HIDDEN_SIZE])
inputs = tf.nn.embedding_lookup(embedding, inputs)
inputs = tf.nn.dropout(inputs, keepprb)

# ======搭建lstm结构=====
lstm = tf.contrib.rnn.LSTMCell(HIDDEN_SIZE, state_is_tuple=False)
lstm = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keepprb)
cell = tf.contrib.rnn.MultiRNNCell([lstm] * HIDDEN_LAYERS)
state = cell.zero_state(BATCH_SIZE, tf.float32)

outputs, stats = tf.nn.dynamic_rnn(cell, inputs, initial_state=state)

state = cell.zero_state(BATCH_SIZE, tf.float32)
print(stats)

