# -----------------------------------------------------------------------------------------------------
'''
@author:	hongwen sun
&usage:		Keras练习之全连接神经网络
'''
# -----------------------------------------------------------------------------------------------------
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, Dropout
from keras import regularizers
from keras.optimizers import Adam
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist/", one_hot = True, reshape=False)
xs = mnist.train.images
ys = mnist.train.labels


# =============定义网络结构==============
inputs = Input(shape=(28, 28, 1))
h1 = Conv2D(64, 3, padding='same', activation='relu')(inputs)
h1 = MaxPooling2D()(h1)
h2 = Conv2D(32, 3, padding='same', activation='relu')(h1)
h2 = MaxPooling2D()(h2)
h3 = Conv2D(16, 3, padding='same', activation='relu')(h2)
h3 = Reshape((16 * 7 * 7,))(h3)
outputs = Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(h3)
model = Model(input=inputs, output=outputs)

# ============训练所需损失函数==========
opt = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# ================开始训练==============
model.fit(x=xs, y=ys, validation_split=0.1, epochs=1)

model.save('k_dnn.h5')