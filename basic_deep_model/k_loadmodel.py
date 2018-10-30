# -----------------------------------------------------------------------------------------------------
'''
@author:	hongwen sun
&usage:		Keras练习之读取模型
'''
# -----------------------------------------------------------------------------------------------------

from keras.models import load_model
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist/", one_hot = True, reshape=False)
xs = mnist.test.images
ys = mnist.test.labels

#=============读取模型=============
model = load_model('k_dnn.h5')

#=============评估模型=============
evl = model.evaluate(x=xs, y=ys)
evl_name = model.metrics_names
for i in range(len(evl)):
	print(evl_name[i], ':\t',evl[i])

