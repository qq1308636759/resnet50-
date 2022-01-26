from tensorflow.keras.models import *
import pandas as pd
import cv2
import numpy as np
def model():
    model = load_model('resnet.h5')
    return model

def read(path):

    img = cv2.imread(path)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
    img = img.reshape(1, 256, 256, 3)
    return img
def pre(model,img):
    pred = model.predict(img)
    y = np.argmax(pred, axis=-1)
    labels= {0:'其他垃圾',1:'厨余垃圾',2:'可回收物',3:'有害垃圾'}
    y = pd.DataFrame(y)
    y[0]=y[0].map(labels)
    y = y.values.flatten()
    print('该垃圾为:',y)
    return y

if __name__ == '__main__':
    path = r'D:\user\Resnet\train_data\img_4.jpg'
    img = read(path)
    model = model()
    pred = pre(model,img)
    print(pred)
