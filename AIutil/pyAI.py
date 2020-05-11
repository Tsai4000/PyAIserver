import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import keras
import os
import cv2
import base64
from keras.layers import Conv2D  # Convolution Operation
from keras.layers import MaxPooling2D # Pooling
from keras.layers import Flatten
from keras.layers import Dense # Fully Connected Networks
from keras.layers import Dropout
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from PIL import Image

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior() 

def predict(inpImg):
    model = load_model('./AIutil/weight/weit.h5', compile=True)
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

    imgdata = base64.b64decode(inpImg)
    img_array = np.fromstring(imgdata, np.uint8) # 轉換np序列
    img = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)  # 轉換Opencv格式
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    matrix = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
    matrix = matrix.reshape(-1, 28, 28, 1)
    print(model.predict(matrix).argmax())
   
    return model.predict(matrix).argmax()

# result = model.predict(testdf)
# a = np.array(result).astype(np.int)
# testGuess = np.argmax(a, axis=1)
# with open('submission.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["id", "label"])
#     aa=testGuess.tolist()
# #     writer.writerow(range(len(aa)), aa)
#     for index in range(len(aa)):
# #         print(index, aa[index])
#         writer.writerow([index, aa[index]])