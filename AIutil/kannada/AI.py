# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior() 
# import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import keras
from keras.layers import Conv2D  # Convolution Operation
from keras.layers import MaxPooling2D # Pooling
from keras.layers import Flatten
from keras.layers import Dense # Fully Connected Networks
from keras.layers import Dropout
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from PIL import Image

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#         if filename == "train.csv":
#             with open(os.path.join(dirname, filename), newline = '') as csvF:
#                 rows = csv.DictReader(csvF)
#                 for row in rows:
#                     print(row)


# filename = "/kaggle/input/Kannada-MNIST/train.csv"
# df = pd.read_csv(filename)
# df = df.sample(frac=1) #打亂 frac為比例
df = np.loadtxt("/kaggle/input/Kannada-MNIST/train.csv",dtype=str,delimiter=',')
df = np.delete(df,0,axis=0)
np.random.shuffle(df)
label = df[:,:1]
# print(label)
df = np.delete(df,0,axis=1)
df = df.reshape((-1,28,28,1))
trainSet = df
# print(np.array(trainSet).max())
# trainSet = np.true_divide(trainSet, 255)
trainSet = np.array(trainSet).astype(np.float)
trainLabel = label
trainLabel = trainLabel.reshape((60000))
a = np.array(trainLabel).astype(np.int)
trainLabel = np.zeros((a.size, a.max()+1))
trainLabel[np.arange(a.size),a] = 1
# trainLabel = np.eye(10)[np.array(trainLabel)]

testdf = np.loadtxt("/kaggle/input/Kannada-MNIST/test.csv",dtype=str,delimiter=',')
testdf = np.delete(testdf,0,axis=0)
# np.random.shuffle(testdf)
# testLabel = testdf[:,:1]
testdf = np.delete(testdf,0,axis=1)
# print(testdf.shape)

testdf = testdf.reshape((-1,28,28,1))


digdf = np.loadtxt("/kaggle/input/Kannada-MNIST/Dig-MNIST.csv",dtype=str,delimiter=',')
digdf = np.delete(digdf,0,axis=0)
np.random.shuffle(digdf)
print(digdf.shape)
digLabel = digdf[:,:1]
print(digLabel.shape)
digLabel = digLabel.reshape((10240))
a = np.array(digLabel).astype(np.int)
digLabel = np.zeros((a.size, a.max()+1))
digLabel[np.arange(a.size),a] = 1
digdf = np.delete(digdf,0,axis=1)
digdf = digdf.reshape((-1,28,28,1))


model=keras.Sequential()  
model.add(Conv2D(16, (5,5), input_shape = (28, 28, 1), activation = 'relu', padding='same'))
model.add(Conv2D(16, (3,3), input_shape = (28, 28, 1), activation = 'relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32, (5,5), input_shape = (28, 28, 1), activation = 'relu', padding='same'))
model.add(Conv2D(32, (3,3), input_shape = (28, 28, 1), activation = 'relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(64, (5,5),input_shape=(28,28,1), activation='relu', padding='same'))
model.add(Conv2D(64, (3,3),input_shape=(28,28,1), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128, (3,3),input_shape=(28,28,1), activation='relu', padding='same'))
model.add(Conv2D(128, (3,3),input_shape=(28,28,1), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(256, (1,1),input_shape=(28,28,1), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(256, (1,1),input_shape=(28,28,1), activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(units = 256, activation = 'relu'))
model.add(Dropout(0.5))  
# model.add(Dense(units = 1024, activation = 'relu'))
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dense(units = 10, activation = 'softmax'))

model.summary()  
opt = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
model.compile(optimizer = opt, loss = 'mean_squared_error', metrics = ['accuracy'])

# print(trainSet.shape)
checkpoint = keras.callbacks.ModelCheckpoint("save.h5",verbose=1,save_best_only=True)
rate = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.75,
                              patience=3, min_lr=0.00001, verbose=1)
train_history = model.fit(x=trainSet,  
                          y=trainLabel, validation_split=0.2,  
                          epochs=80, batch_size=600, verbose=2, callbacks=[checkpoint, rate])

# model.save("save.h5")

# model = load_model('/kaggle/working/save.h5', compile=True)
# model = load_model('/kaggle/input/weight1/weit.h5', compile=True)
# model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

score = model.evaluate(digdf, digLabel, verbose=0)
print(score[0], score[1])
# b = np.zeros((a.size, 10))
# b[np.arange(a.size),a] = 1
# print('test after load: ', a)
# print('ss: ', testGuess)



result = model.predict(testdf)
a = np.array(result).astype(np.int)
testGuess = np.argmax(a, axis=1)
with open('submission.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["id", "label"])
    aa=testGuess.tolist()
#     writer.writerow(range(len(aa)), aa)
    for index in range(len(aa)):
#         print(index, aa[index])
        writer.writerow([index, aa[index]])