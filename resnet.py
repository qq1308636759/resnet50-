import pandas as pd
import numpy as np
import os
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.applications import Xception,ResNet50,MobileNetV2,VGG16
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import keras
from keras import layers, optimizers, models
from keras.layers import *
from sklearn.metrics import confusion_matrix, classification_report
# 首先读取数据分析样本和标签格式,制作样本标签文件
# --------------------------------------------------------------------
# filenames = os.listdir(r'D:\user\Resnet\label')
# filenames.sort(key=lambda x:int(x.split('img_')[1].split('.txt')[0]))
# print(filenames)
# label = []
# text = []
# label = pd.DataFrame(label)
# for i in range(len(filenames)):
#     path = 'label/' + filenames[i]
#     df1 = pd.read_csv(path,header=None)
#     label = label.append(df1)
# label.columns = ['name','label']
#
# label.to_csv('label.txt')
# print(label)

# ----------------------------------------------------------------------


# 图像的读取保存npy文件
# # -----------------------------------------------------------------------
# X = []  # 图片灰度值
# Y = []  # 分类标签
# data = pd.read_csv('label.txt')
# name = data['name']
# a=data['label']
# for i in range(len(data)):
#     if a[i]<6:
#         Y.append(0)
#     elif 5<a[i]<14:
#         Y.append(1)
#     elif 13<a[i]<37:
#         Y.append(2)
#     elif 36<a[i]<40:
#         Y.append(3)
#     else:
#         break
# for i in range(len(data)):
#       path = 'train_data/' + name[i]
#       Images = cv2.imread(path)
#       print(path)
#       image = cv2.resize(Images, (256,256), interpolation=cv2.INTER_LINEAR)
#       X.append((image))
# np.save('x.npy',X)
# np.save('y.npy',Y)
# ---------------------------------------------------------------------------

#标签独热码的转化和数据集的划分
# -----------------------------------------------------------------------------
X = np.load('x.npy')
Y = np.load('y.npy')
Y = to_categorical(Y, 4)
x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size=0.5, random_state=1)
# -------------------------------------------------------------------------------

# 模型搭建和配置训练
# -----------------------------------------------------------------------------
conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(layers.Dense(4, activation='softmax'))
conv_base.trainable = False
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0001), metrics=['acc'])
early_stop = keras.callbacks.EarlyStopping(monitor='val_acc', patience=6)
model_checkpoint = ModelCheckpoint('model.h5', monitor='val_acc', verbose=1, save_best_only=True)

history = model.fit(x_train, y_train, epochs=10, batch_size=200,validation_data=(x_test,y_test),callbacks=[early_stop,model_checkpoint])



model=load_model('moedel.h5')




#垃圾识别
# ----------------------------
path='垃圾图片的路径'
pred = model.predict(x_test)
y = np.argmax(pred, axis=-1)
y_test = np.argmax(y_test, axis=-1)
print(confusion_matrix(y_test, y))
print(classification_report(y_test, y))
cm = confusion_matrix(y_test, y)
print(cm)
plt.imshow(cm, cmap=plt.cm.BuPu)
# --------------------------------

# 结果显示
# -------------------------
indices = range(len(cm))
label_name= ['其他垃圾', '厨余垃圾', '可回收物', '有害垃圾']
ax = plt.gca()
plt.xticks(indices,label_name,fontsize=8)
ax.xaxis.set_ticks_position("top")
plt.yticks(indices, label_name,fontsize=8)

plt.colorbar()

plt.xlabel('预测值')
plt.ylabel('真实值')
plt.title('混淆矩阵')
#
# plt.rcParams两行是用于解决标签不能显示汉字的问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#
# 显示数据
for first_index in range(len(cm)):    #第几行
    for second_index in range(len(cm[first_index])):    #第几列
        plt.text(first_index, second_index, cm[first_index][second_index],fontdict={'size':6})
# 在matlab里面可以对矩阵直接imagesc(confusion)
# 显示
plt.show()
# ---------------------------------------------------------------------------