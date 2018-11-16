# ========读取原始数据========
with open('cmn.txt', 'r', encoding='utf-8') as f:
    data = f.read()
data = data.split('\n')
data = data[:10000]
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



import numpy as np
# 利用字典，映射数据
en_num_data = [[en2id[en] for en in line] for line in en_data]
ch_num_data = [[ch2id['<GO>']] + [ch2id[ch] for ch in line] for line in ch_data]
de_num_data = [[ch2id[ch] for ch in line] + [ch2id['<EOS>']] for line in ch_data]

print('char:', en_data[1])
print('index:', en_num_data[1])


en_maxlength = max([len(line) for line in en_num_data])
de_maxlength = max([len(line) for line in ch_num_data])



# 设计数据生成器
def batch_data(en_num_data, ch_num_data, de_num_data, batch_size):
    batch_num = len(en_num_data) // batch_size
    for i in range(batch_num):
        begin = i * batch_size
        end = begin + batch_size
        encoder_inputs = en_num_data[begin:end]
        decoder_inputs = ch_num_data[begin:end]
        decoder_targets = de_num_data[begin:end]
        encoder_lengths = [len(line) for line in encoder_inputs]        
        decoder_lengths = [len(line) for line in decoder_inputs]
        encoder_max_length = max(encoder_lengths)
        decoder_max_length = max(decoder_lengths)
        encoder_inputs = np.array([data + [en2id['<PAD>']] * (encoder_max_length - len(data)) for data in encoder_inputs]).T
        decoder_inputs = np.array([data + [en2id['<PAD>']] * (decoder_max_length - len(data)) for data in decoder_inputs]).T
        decoder_targets = np.array([data + [en2id['<PAD>']] * (decoder_max_length - len(data)) for data in decoder_targets])
        mask = decoder_targets > 0
        target_weights = np.ma.array(decoder_targets,mask=mask).astype(np.float32)
        yield encoder_inputs, decoder_inputs, decoder_targets, target_weights, encoder_lengths, decoder_lengths
              



import tensorflow as tf

max_encoder_seq_length = en_maxlength
max_decoder_seq_length = de_maxlength
keepprb = 0.9

EN_VOCAB_SIZE = len(en2id)
CH_VOCAB_SIZE = len(ch2id)

HIDDEN_LAYERS = 1
HIDDEN_SIZE = 512

learning_rate = 0.001

BATCH_SIZE = 8
BATCH_NUMS = len(ch_num_data) // BATCH_SIZE
MAX_GRAD_NORM = 1

EPOCHS = 50



encoder_inputs = tf.placeholder(tf.int32, [None, BATCH_SIZE])
decoder_inputs = tf.placeholder(tf.int32, [None, BATCH_SIZE])
decoder_targets = tf.placeholder(tf.int32, [BATCH_SIZE, None])
target_weights = tf.placeholder(tf.float32, [BATCH_SIZE, None])
source_sequence_length = tf.placeholder(tf.int32, [BATCH_SIZE,])
decoder_lengths = tf.placeholder(tf.int32, [BATCH_SIZE,])

keepprb = tf.placeholder(tf.float32)



# encoder
with tf.name_scope('embedding_encoder'):
	encoder_embedding = tf.get_variable('embedding_encoder', [EN_VOCAB_SIZE, HIDDEN_SIZE])
	encoder_emb = tf.nn.embedding_lookup(encoder_embedding, encoder_inputs)
	encoder_emb = tf.nn.dropout(encoder_emb, keepprb)

    
# decoder
with tf.name_scope('embedding_decoder'):
	decoder_embedding = tf.get_variable('embedding_decoder', [CH_VOCAB_SIZE, HIDDEN_SIZE])
	decoder_emb = tf.nn.embedding_lookup(decoder_embedding, decoder_inputs)
	decoder_emb = tf.nn.dropout(decoder_emb, keepprb)



# encoder
with tf.variable_scope('encoder'):
	encoder_lstm = tf.contrib.rnn.LSTMCell(HIDDEN_SIZE, state_is_tuple=True)
	encoder_lstm = tf.contrib.rnn.DropoutWrapper(encoder_lstm, output_keep_prob=keepprb)
	encoder_cell = tf.contrib.rnn.MultiRNNCell([encoder_lstm for _ in range(HIDDEN_LAYERS)])
	initial_state = encoder_cell.zero_state(BATCH_SIZE, tf.float32)
	encoder_outputs, final_state = tf.nn.dynamic_rnn(encoder_cell, encoder_emb, sequence_length=source_sequence_length, 
                                       time_major=True, initial_state=initial_state)



attention_states = tf.transpose(encoder_outputs, [1, 0, 2])
# Create an attention mechanism
attention_mechanism = tf.contrib.seq2seq.LuongAttention(HIDDEN_SIZE, attention_states, memory_sequence_length=source_sequence_length)


from tensorflow.python.layers.core import Dense

# decoder cell
with tf.variable_scope('decoder_cell'):
    decoder_lstm = tf.contrib.rnn.LSTMCell(HIDDEN_SIZE, state_is_tuple=True)
    decoder_lstm = tf.contrib.rnn.DropoutWrapper(decoder_lstm, output_keep_prob=keepprb)
    decoder_cell = tf.contrib.rnn.MultiRNNCell([decoder_lstm] * HIDDEN_LAYERS)
    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size=HIDDEN_SIZE)
    # Helper
    projection_layer = Dense(CH_VOCAB_SIZE, use_bias=False)
    with tf.variable_scope('helper'):
        helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb, decoder_lengths, time_major=True)
    init_state = decoder_cell.zero_state(BATCH_SIZE, tf.float32).clone(cell_state=final_state)
    decoder_cell = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, init_state, output_layer=projection_layer)
    print(final_state)
    print(init_state)
    print(projection_layer)
    outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder_cell)

    logits = outputs.rnn_output




with tf.variable_scope('optimizer'):
    # ======计算损失=======
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_targets, logits=logits)
    cost = (tf.reduce_sum((loss * target_weights) / BATCH_SIZE))

    # =============优化算法==============
    # =============学习率衰减==============
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(learning_rate, global_step, BATCH_NUMS, 0.99, staircase=True)

                # =======通过clip_by_global_norm()控制梯度大小======
    trainable_variables = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, trainable_variables), MAX_GRAD_NORM)
    opt = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(grads, trainable_variables))

		# ==============预测输出=============
predict = tf.argmax(logits[0], 1)

# 保存模型
saver = tf.train.Saver()
with tf.Session() as sess:
	writer = tf.summary.FileWriter('logs/tensorboard', tf.get_default_graph())
	sess.run(tf.global_variables_initializer())
	for k in range(EPOCHS):
		total_loss = 0.
		data_generator = batch_data(en_num_data, ch_num_data, de_num_data, BATCH_SIZE)
		for i in range(BATCH_NUMS):
			en_input, de_input, de_tg, tg_weight, en_len, de_len = next(data_generator)
			feed = {encoder_inputs: en_input, decoder_inputs: de_input, decoder_targets: de_tg, target_weights: tg_weight, source_sequence_length: en_len, decoder_lengths: de_len, keepprb: 0.8}
			costs, _ = sess.run([cost, opt], feed_dict=feed)
			total_loss += costs
			if (i+1) % 50 == 0:
				print('epochs:', k + 1, 'iter:', i + 1, 'cost:', total_loss / i + 1)
				#print('predict:', sess.run(predict[0], feed_dict=feed))
				print('text:', ''.join([id2ch[i] for i in sess.run(predict, feed_dict=feed)]))
				print('label:', ''.join([id2ch[i] for i in de_tg[0]]))
                
	saver.save(sess, './checkpoints/lstm.ckpt')

writer.close()