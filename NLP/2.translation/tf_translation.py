# ========读取原始数据========
with open('cmn.txt', 'r', encoding='utf-8') as f:
    data = f.read()
data = data.split('\n')
data = data[:1000]
print(data[-5:])


# 分割英文数据和中文数据
en_data = [line.split('\t')[0] for line in data]
ch_data = [line.split('\t')[1] for line in data]
print('英文数据:\n', en_data[:10])
print('\n中文数据:\n', ch_data[:10])


# 特殊字符
SOURCE_CODES = ['<PAD>', '<UNK>']
TARGET_CODES = ['<PAD>', '<EOS>', '<UNK>', '<GO>']  # 在target中，需要增加<GO>与<EOS>特殊字符

# 分别生成中英文字典
en_vocab = set(''.join(en_data))
id2en = SOURCE_CODES + list(en_vocab)
en2id = {c:i for i,c in enumerate(id2en)}

# 分别生成中英文字典
ch_vocab = set(''.join(ch_data))
id2ch = TARGET_CODES + list(ch_vocab)
ch2id = {c:i for i,c in enumerate(id2ch)}

print('\n英文字典:\n', en2id)
print('\n中文字典共计\n:', ch2id)


# 利用字典，映射数据
en_num_data = [[en2id[en] for en in line] for line in en_data]
ch_num_data = [[ch2id['<GO>']] + [ch2id[ch] for ch in line] + [ch2id['<EOS>']] for line in ch_data]
de_num_data = [[ch2id[ch] for ch in line] + [ch2id['<EOS>']] for line in ch_data]

print('char:', en_data[1])
print('index:', en_num_data[1])

en_maxlength = max([len(line) for line in en_num_data])
ch_maxlength = max([len(line) for line in ch_num_data])

# 文本数据转化为数字数据
en_num_data = [data + [en2id['<PAD>']] * (en_maxlength - len(data)) for data in en_num_data]
ch_num_data = [data + [en2id['<PAD>']] * (ch_maxlength - len(data)) for data in ch_num_data]
de_num_data = [data + [en2id['<PAD>']] * (ch_maxlength - len(data)) for data in de_num_data]


# 设计数据生成器
def batch_data(en_num_data, ch_num_data, de_num_data, batch_size):
    batch_num = len(en_num_data) // batch_size
    for i in range(batch_num):
        begin = i * batch_size
        end = begin + batch_size
        x = en_num_data[begin:end]
        y = ch_num_data[begin:end]
        z = de_num_data[begin:end]
        yield x, y, z


import tensorflow as tf

max_encoder_seq_length = en_maxlength
max_decoder_seq_length = ch_maxlength
keepprb = 0.9

EN_VOCAB_SIZE = len(en2id)
CH_VOCAB_SIZE = len(ch2id)

HIDDEN_LAYERS = 2
HIDDEN_SIZE = 512

learning_rate = 0.003

BATCH_SIZE = 8
BATCH_NUMS = len(ch_num_data) // BATCH_SIZE
MAX_GRAD_NORM = 1

EPOCHS = 100



encoder_inputs = tf.placeholder(tf.int32, [BATCH_SIZE, max_encoder_seq_length])
decoder_inputs = tf.placeholder(tf.int32, [BATCH_SIZE, max_decoder_seq_length])
targets = tf.placeholder(tf.int32, [BATCH_SIZE, max_decoder_seq_length])
keepprb = tf.placeholder(tf.float32)


with tf.name_scope('embedding_encoder'):
	encoder_embedding = tf.get_variable('embedding_encoder', [EN_VOCAB_SIZE, HIDDEN_SIZE])
	encoder_emb = tf.nn.embedding_lookup(encoder_embedding, encoder_inputs)
	encoder_emb = tf.nn.dropout(encoder_emb, keepprb)


# encoder
with tf.variable_scope('encoder'):
	encoder_lstm = tf.contrib.rnn.LSTMCell(HIDDEN_SIZE, state_is_tuple=True)
	encoder_lstm = tf.contrib.rnn.DropoutWrapper(encoder_lstm, output_keep_prob=keepprb)
	encoder_cell = tf.contrib.rnn.MultiRNNCell([encoder_lstm] * HIDDEN_LAYERS)
	initial_state = encoder_cell.zero_state(BATCH_SIZE, tf.float32)
	_, final_state = tf.nn.dynamic_rnn(encoder_cell, encoder_emb, initial_state=initial_state)


with tf.name_scope('embedding_decoder'):
	decoder_embedding = tf.get_variable('embedding_decoder', [CH_VOCAB_SIZE, HIDDEN_SIZE])
	decoder_emb = tf.nn.embedding_lookup(decoder_embedding, decoder_inputs)
	decoder_emb = tf.nn.dropout(decoder_emb, keepprb)


# decoder
with tf.variable_scope('decoder'):
	decoder_lstm = tf.contrib.rnn.LSTMCell(HIDDEN_SIZE, state_is_tuple=True)
	decoder_lstm = tf.contrib.rnn.DropoutWrapper(decoder_lstm, output_keep_prob=keepprb)
	decoder_cell = tf.contrib.rnn.MultiRNNCell([decoder_lstm] * HIDDEN_LAYERS)
	outputs, _ = tf.nn.dynamic_rnn(decoder_cell, decoder_emb, initial_state=final_state)
	outputs = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])


w = tf.get_variable('outputs_weight', [HIDDEN_SIZE, CH_VOCAB_SIZE])
b = tf.get_variable('outputs_bias', [CH_VOCAB_SIZE])
logits = tf.matmul(outputs, w) + b

		# ======计算损失=======
loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [tf.reshape(targets, [-1])], 
														[tf.ones([BATCH_SIZE * max_decoder_seq_length], dtype=tf.float32)])
cost = tf.reduce_sum(loss) / BATCH_SIZE

		# =============优化算法==============
          # =============学习率衰减==============
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(learning_rate, global_step, BATCH_NUMS, 0.99, staircase=True)

			# =======通过clip_by_global_norm()控制梯度大小======
trainable_variables = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, trainable_variables), MAX_GRAD_NORM)
opt = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(grads, trainable_variables))

		# ==============预测输出=============
predict = tf.reshape(tf.argmax(logits, 1), [-1, max_decoder_seq_length])


# 保存模型
saver = tf.train.Saver()
with tf.Session() as sess:
	writer = tf.summary.FileWriter('logs/tensorboard', tf.get_default_graph())
	sess.run(tf.global_variables_initializer())
	for k in range(EPOCHS):
		total_loss = 0.
		data_generator = batch_data(en_num_data, ch_num_data, de_num_data, BATCH_SIZE)
		for i in range(BATCH_NUMS):
			en_batch, ch_batch, de_batch = next(data_generator)
			feed = {encoder_inputs: en_batch, decoder_inputs: ch_batch, targets: de_batch, keepprb: 0.8}
			costs, _ = sess.run([cost, opt], feed_dict=feed)
			total_loss += costs
			if (i+1) % 50 == 0:
				print('epochs:', k + 1, 'iter:', i + 1, 'cost:', total_loss / i + 1)
				#print('predict:', sess.run(predict[0], feed_dict=feed))
				print('text:', ''.join([id2ch[i] for i in sess.run(predict[0], feed_dict=feed) if(i != 0 and i != 1)]))
				print('label:', ''.join([id2ch[i] for i in de_batch[0] if(i != 0 and i != 1)]))
                
	saver.save(sess, './checkpoints/lstm.ckpt')

writer.close()


