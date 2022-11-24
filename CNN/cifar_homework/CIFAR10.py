import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train),(x_test,y_test) = cifar10.load_data()

x_train = x_train.astype('float32') /255.0
x_test = x_test.astype('float32') /255.0

#建立Sequential线性堆叠模型
model =tf.keras.models.Sequential()
#第1个卷积层
model.add(tf.keras.layers.Conv2D(filters = 32,kernel_size = (3,3),input_shape = (32, 32, 3),activation ='relu',padding ='same'))
#防止过拟合
model.add(tf.keras.layers.Dropout(rate=0.3))
#第1个池化层
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
#第2个卷积层
model.add(tf.keras.layers.Conv2D(filters= 64,kernel_size =(3,3),activation ='relu',padding ='same'))
#防止过拟合
model.add(tf.keras.layers.Dropout(rate=0.3))
#第2个池化层
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# 平坦层
model.add(tf.keras.layers.Flatten())
#添加输出层
model.add(tf.keras.layers.Dense(10, activation ='softmax'))
#模型摘要
model.summary()
#设置训练参数
train_epochs = 20 #训练轮数
batch_size = 100 #单词次训练样本数
#定义训练模式
model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy' , metrics=['accuracy'])
#训练模型
train_history=model.fit(x_train, y_train, validation_split = 0.2, epochs=train_epochs, batch_size= batch_size, verbose=2)
#定义可视化函数
def visu_train_history(train_history,train_metric,validation_metric):
    plt.plot(train_history.history[train_metric])
    plt.plot(train_history.history[validation_metric])
    plt.title('Train History')
    plt.ylabel(train_metric)
    plt.xlabel('epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()
visu_train_history(train_history,'loss','val_loss')
visu_train_history(train_history,'accuracy','val_accuracy')
#评估模型
test_loss, test_acc=model.evaluate(x_test, y_test, verbose=2)
print('Test accuracy:', test_acc)