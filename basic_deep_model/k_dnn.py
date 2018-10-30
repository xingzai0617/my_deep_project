# -----------------------------------------------------------------------------------------------------
'''
@author:	hongwen sun
&usage:		Keras练习之全连接神经网络
'''
# -----------------------------------------------------------------------------------------------------
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras import regularizers
from keras.optimizers import Adam

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist/", one_hot = True)
xs = mnist.train.images
ys = mnist.train.labels

# =============定义网络结构==============
inputs = Input(shape=(784,))
h1 = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(inputs)
h1 = Dropout(0.2)(h1)
h2 = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(h1)
h2 = Dropout(0.2)(h2)
h3 = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(h2)
h3 = Dropout(0.2)(h3)
outputs = Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(h3)
model = Model(input=inputs, output=outputs)

# ============训练所需损失函数==========
opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# ================开始训练==============
model.fit(x=xs, y=ys, validation_split=0.1, epochs=4)

model.save('k_dnn.h5')