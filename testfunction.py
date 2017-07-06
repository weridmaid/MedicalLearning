#coding:utf-8
'''
Created on 2015-9-12
@author: zzq2015
'''
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import scipy.io as sio
import numpy as np

model = Sequential()
model.add(Dense(4, 200, init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(200, 100, init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(100, 50, init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(50, 20, init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(20, 3, init='uniform'))
model.add(Activation('softmax'))


model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

matfn=u'/media/zzq2015/学习/python/da/kerasTrain.mat'
data=sio.loadmat(matfn)
data = np.array(data.get('iris_train'))
trainDa = data[:80,:4]
trainBl  = data[:80,4:]
testDa = data[80:,:4]
testBl  = data[80:,4:]

model.fit(trainDa, trainBl, nb_epoch=80, batch_size=20)
print (model.evaluate(testDa, testBl, show_accuracy=True))
print (model.predict_classes(testDa))
print ('真实标签：\n')
print (testBl)