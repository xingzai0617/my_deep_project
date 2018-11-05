
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


# ========================生成字典=====================
import os
import json

if os.path.exists('vocab.json'):
	with open('vocab.json', 'r', encoding='utf-8') as f:
		id2char = json.load(f)	
else:
	vocab = set(data)
	id2char = list(vocab)
	with open('vocab.json', 'w', encoding='utf-8') as f:
		json.dump(id2char, f)

char2id = {c:i for i,c in enumerate(id2char)}
print('字典长度:', len(id2char))


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
data_batch = data_generator(numdata, 4, 100)
x, y = next(data_batch)
print('input data:', x[0], '\noutput data:', y[0])

# 将y转化为onehot格式
def onehot(y):
	y_onehot = np.zeros(shape=(y.shape[0], y.shape[1], VOCAB_SIZE))
	for i in range(y.shape[0]):
		for j in range(y.shape[1]):
			y_onehot[i][j][y[i][j]] = 1
	return y_onehot


# ======================================keras model==================================
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, Embedding
from keras import regularizers
from keras.optimizers import Adam
import numpy as np
# =======预定义模型参数========
VOCAB_SIZE = len(id2char)
EPOCHS = 50
BATCH_SIZE = 8
TIME_STEPS = 100
BATCH_NUMS = len(numdata) // (BATCH_SIZE * TIME_STEPS)
HIDDEN_SIZE = 512
HIDDEN_LAYERS = 3
MAX_GRAD_NORM = 1
learning_rate = 0.003

inputs = Input(shape=(None,))
emb_inp = Embedding(output_dim=HIDDEN_SIZE, input_dim=VOCAB_SIZE, input_length=None)(inputs)
h1 = LSTM(HIDDEN_SIZE, activation='relu', return_sequences=True, dropout=0.2)(emb_inp)
h2 = LSTM(HIDDEN_SIZE, activation='relu', return_sequences=True, dropout=0.2)(h1)
output = Dense(VOCAB_SIZE, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(h2)
model = Model(inputs=inputs, outputs=output)

# ============训练所需损失函数==========
opt = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

data_batch = data_generator(numdata, BATCH_SIZE, TIME_STEPS)
for i in range(100):
	x, y = next(data_batch)
	y_onehot = onehot(y)
	model.fit(x, y_onehot)

# ================预测=================
x_initial = np.zeros((1,1), dtype=int) + 8
x_test = x_initial
for i in range(100):
	y_pre = model.predict(x_test)
	index = np.argmax(y_pre, 2)
	x_test = np.concatenate((x_initial, index), axis=1)
x_test = np.reshape(x_test, (-1,))
print(''.join([id2char[i] for i in x_test]))