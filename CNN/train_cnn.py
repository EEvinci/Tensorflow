# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 10:07:21 2022

@author: wjcy19870122
"""

import os
import os.path as osp
import tensorflow as tf

def create_cnn():
    '''
    定义一个CNN分类模型，该模型的特征提取器（feature extractor）包含
    两层卷积，每层卷积后面通过Maxpooling降低分辨率；分类器（classifier）
    由一层全连接层（fully-connected layers，FC）+一层输出层构成。
    '''
    #该CNN模型为串行模型
    model = tf.keras.models.Sequential()
    
    #=================== feature extractor ====================
    #第一层卷积需要指定输入图像的shape为[batchsize，32，32，3]
    #输出特征shape：[batchsize，32, 32, 32]
    model.add(tf.keras.layers.Conv2D(filters=32,
                                     kernel_size=(3,3),
                                     input_shape=(32, 32, 3),
                                     activation='relu',
                                     padding='same'))   
    
    
    #将特征图缩小一半，输出shape：[batchsize，16, 16, 32]
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    #第二层卷积输出特征shape：[batchsize，16, 16, 64]
    model.add(tf.keras.layers.Conv2D(filters=64,
                                     kernel_size=(3,3),
                                     input_shape=(32, 32, 3),
                                     activation='relu',
                                     padding='same'))
    
    
    #将特征图缩小一半，输出shape：[batchsize，8, 8, 64]
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    #将2D的特征图变为向量，输出shape：[batchsize, 4096]
    model.add(tf.keras.layers.Flatten())
    
    #===========--------- classifier ==========================
    #第一层FC， 输出shape:[batchsize, 1024]
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    
    #随机将第一层FC的30%weights值设置为0，缓解过拟合
    #输出shape：[batchsize，1024]
    model.add(tf.keras.layers.Dropout(rate=0.3))
    
    #第二层FC层作为输出层，10个类别的softmax概率
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    
    return model

#主函数，导入数据集->实例化模型->定义训练超参数->定义优化器->迭代训练并保存模型
if __name__ == '__main__':
    #======================导入cifar10数据集 =====================
    cifar10 = tf.keras.datasets.cifar10
    #C:\Users\XXXX\.keras\datasets下寻找数据集，如没有，则会自动下载
    (x_train, y_train),(x_test, y_test) = cifar10.load_data()
    
    #通过打印shape查看数据集基本信息
    print('train images:', x_train.shape) #[50000, 32, 32, 3]
    print('train labels:', y_train.shape) #[50000, 1]
    print('test images:', x_test.shape) #[10000, 32, 32, 3]
    print('test labels:', y_test.shape) #[10000, 1]
    
    #===================== 实例化CNN模型 ========================
    model = create_cnn()
    
    print('模型结构为：')
    model.summary()
    
    #===================== 定义训练超参数 =======================   
    model.compile(optimizer='adam', #使用adam优化器
                  loss='sparse_categorical_crossentropy', #交叉熵损失
                  metrics=['accuracy']) #准确率作为评估性能指标
    
    #=============== 开始迭代训练并保存模型参数=================
    #首先创建模型保存的目录    
    save_root = './checkpoints'
    if not osp.exists(save_root): os.mkdir(save_root)
    
    #保存模型的文件名，占位符会被自动替换为epoch值和指标值
    checkpoint_file = osp.join(save_root,'Cifar10.{epoch:02d}-{val_loss:4f}.h5')
    #通过调用该回调函数让模型保存参数
    call_backs = [
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_file,
                                           save_weights_only=True, #只保存参数，不保存网络结构 如果要保存网络结构文件会很大
                                           save_freq='epoch')#每次epoch结束才保存模型)
        ]
    train_history = model.fit(x_train, y_train, 
                              validation_split=0.2,
                              epochs=20,#训练5个轮次
                              batch_size=100,#每次喂入模型100张图像
                              callbacks=call_backs,
                              verbose=2)#控制打印信息的level
    
    
                  
        
    