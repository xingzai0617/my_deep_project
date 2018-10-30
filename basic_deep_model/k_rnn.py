# -----------------------------------------------------------------------------------------------------
'''
@author:	hongwen sun
&usage:		Keras练习之全连接神经网络
'''
# -----------------------------------------------------------------------------------------------------
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout
from keras import regularizers
from keras.optimizers import Adam
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist/", one_hot = True)
xs = mnist.train.images
xs = xs.reshape(-1, 28, 28)
ys = mnist.train.labels

# =============定义网络结构==============
inputs = Input(shape=(28, 28))
h1 = LSTM(64, activation='relu', return_sequences=True, dropout=0.2)(inputs)
h2 = LSTM(64, activation='relu', dropout=0.2)(h1)
outputs = Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(h2)
model = Model(input=inputs, output=outputs)

# ============训练所需损失函数==========
opt = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# ================开始训练==============
model.fit(x=xs, y=ys, validation_split=0.1, epochs=1)

model.save('k_dnn.h5')