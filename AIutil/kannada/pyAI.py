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
    model = load_model('./AIutil/kannada/weight/weit.h5', compile=True)
    n, x, y, t = model.get_config()['layers'][0]['config']['batch_input_shape']
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

    imgdata = base64.b64decode(inpImg)
    img_array = np.fromstring(imgdata, np.uint8)
    img = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    matrix = cv2.resize(img, (x, y), interpolation=cv2.INTER_CUBIC)
    matrix = matrix.reshape(-1, x, y, t)
    predictList = model.predict(matrix)
    print(predictList)
    result = predictList.argmax()
    return result, predictList[0][result]
