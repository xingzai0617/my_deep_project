# ========读取原始数据========
with open('cmn.txt', 'r', encoding='utf-8') as f:
    data = f.read()
data = data.split('\n')
data = data[:5000]
print(data[-1:])

# 分割英文数据和中文数据
en_data = [line.split('\t')[0] for line in data]
ch_data = [line.split('\t')[1] for line in data]
print('英文数据:', en_data[:10])
print('中文数据:', ch_data[:10])

# 分别生成中英文字典
en_vocab = set(''.join(en_data))
id2en = ['__PAD__', '__UNK__'] + list(en_vocab)
en2id = {c:i for i,c in enumerate(id2en)}

ch_vocab = set(''.join(ch_data))
id2ch = ['__PAD__', '__UNK__', '__GO__', '__EOS__'] + list(ch_vocab)
ch2id = {c:i for i,c in enumerate(id2ch)}

print('英文字典:\n', en2id)
print('中文字典共计:', len(ch2id))

en_num_data = [[en2id[en] for en in line ] for line in en_data]
ch_num_data = [[ch2id['__GO__']] + [ch2id[ch] for ch in line] for line in ch_data]
de_num_data = [[ch2id[ch] for ch in line] + [ch2id['__EOS__']] for line in ch_data]
print(en_num_data[:5])
print(ch_num_data[:5])
print(de_num_data[:5])

import numpy as np

max_encoder_seq_length = max([len(txt) for txt in en_num_data])
max_decoder_seq_length = max([len(txt) for txt in ch_num_data])
print(max_encoder_seq_length)
print(max_decoder_seq_length)


encoder_input_data = [line + [0] * (max_encoder_seq_length-len(line)) for line in en_num_data]
decoder_input_data = [line + [0] * (max_decoder_seq_length-len(line)) for line in ch_num_data]
decoder_output_data = [line + [0] * (max_decoder_seq_length-len(line)) for line in de_num_data]
decoder_target_data = np.zeros((len(ch_num_data), max_decoder_seq_length, len(ch2id)), dtype='float32')
for i in range(len(ch_num_data)):
    for j in range(max_decoder_seq_length):
        decoder_target_data[i,j,decoder_output_data[i][j]] = 1

print(decoder_target_data.shape)

# =======预定义模型参数========
EN_VOCAB_SIZE = len(en2id)
CH_VOCAB_SIZE = len(ch2id)
EPOCHS = 10
HIDDEN_SIZE = 256

# ======================================keras model==================================
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, Embedding, Masking
from keras import regularizers
from keras.optimizers import Adam
import numpy as np

# ==============encoder=============
encoder_inputs = Input(shape=(None,))
emb_inp = Embedding(output_dim=HIDDEN_SIZE, input_dim=EN_VOCAB_SIZE, mask_zero=True)(encoder_inputs)
encoder_h1, encoder_state_h1, encoder_state_c1 = LSTM(HIDDEN_SIZE, activation='relu', return_sequences=True, return_state=True, dropout=0.2)(emb_inp)
encoder_h2, encoder_state_h2, encoder_state_c2 = LSTM(HIDDEN_SIZE, activation='relu', return_state=True, dropout=0.2)(encoder_h1)
encoder_state = [[encoder_state_h1, encoder_state_c1],[encoder_state_h2, encoder_state_c2]]

# ==============decoder=============
decoder_inputs = Input(shape=(None, ))

emb_target = Embedding(output_dim=HIDDEN_SIZE, input_dim=CH_VOCAB_SIZE, mask_zero=True)(decoder_inputs)
lstm1 = LSTM(HIDDEN_SIZE, activation='relu', return_sequences=True, return_state=True, dropout=0.2)
lstm2 = LSTM(HIDDEN_SIZE, activation='relu', return_sequences=True, return_state=True, dropout=0.2)
decoder_dense = Dense(CH_VOCAB_SIZE, activation='softmax')

decoder_h1, _, _ = lstm1(emb_target, initial_state=encoder_state[0])
decoder_h2, _, _ = lstm2(decoder_h1, initial_state=encoder_state[1])
decoder_outputs = decoder_dense(decoder_h2)

batch_size = 64
epochs = 20

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0)

# Save model
model.save('s2s.h5')

encoder_model = Model(encoder_inputs, [encoder_state_h1, encoder_state_c1,encoder_state_h2, encoder_state_c2])

decoder_state_input_h1 = Input(shape=(HIDDEN_SIZE,))
decoder_state_input_c1 = Input(shape=(HIDDEN_SIZE,))
decoder_state_input_h2 = Input(shape=(HIDDEN_SIZE,))
decoder_state_input_c2 = Input(shape=(HIDDEN_SIZE,))

decoder_h1, state_h1, state_c1 = lstm1(emb_target, initial_state=[decoder_state_input_h1, decoder_state_input_c1])
decoder_h2, state_h2, state_c2 = lstm2(decoder_h1, initial_state=[decoder_state_input_h2, decoder_state_input_c2])
decoder_outputs = decoder_dense(decoder_h2)

decoder_model = Model([decoder_inputs, decoder_state_input_h1, decoder_state_input_c1, decoder_state_input_h2, decoder_state_input_c2], 
                      [decoder_outputs, state_h1, state_c1, state_h2, state_c2])

print(encoder_input_data[1])
print(decoder_input_data[1])
print(decoder_output_data[1])
print(''.join([id2en[i] for i in encoder_input_data[1]]))
print(''.join([id2ch[i] for i in decoder_input_data[1]]))
print(''.join([id2ch[i] for i in decoder_output_data[1]]))

for k in range(50):
    test_data = encoder_input_data[k]
    h1, c1,h2, c2 = encoder_model.predict(test_data)
    condition = True
    outputs = []
    decoder_input = [2]
    while condition:
        output, h1, c1, h2, c2 = decoder_model.predict([decoder_input, h1, c1, h2, c2])
        outputs.append(np.argmax(output))
        decoder_input = [np.argmax(output)]
        if decoder_input == 3 or len(outputs) > 20: condition = False
    print(''.join([id2en[i] for i in test_data]))
    print(''.join([id2ch[i] for i in decoder_output_data[k]]))
    print(''.join([id2ch[i] for i in outputs]))
   