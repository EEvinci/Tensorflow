# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:13:31 2022

@author: wjcy19870122
"""
import tensorflow as tf

def VGG16(input_shape=(150, 150, 3), n_classes=6):
    '''
    定义一个VGG16分类模型，该模型的特征提取器（feature extractor）包含
    5组卷积；分类器（classifier）包含两层全连接层（fully-connected layers，FC）+一层输出层构成。
    '''
    model = tf.keras.models.Sequential()
    #group 1
    model.add(tf.keras.layers.Conv2D(filters=16,
                                            kernel_size=(3,3),
                                            input_shape=input_shape,
                                            activation='relu',
                                            padding='same',
                                            name='g1_conv1'))
    
    model.add(tf.keras.layers.Conv2D(filters=16,
                                            kernel_size=(3,3),
                                            activation='relu',
                                            padding='same',
                                            name='g1_conv2'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                           name='g1_maxpool'))
    #group 2
    model.add(tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=(3,3),
                                            activation='relu',
                                            padding='same',
                                            name='g2_conv1'))
    model.add(tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=(3,3),
                                            activation='relu',
                                            padding='same',
                                            name='g2_conv2'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                           name='g2_maxpool'))
    #group 3
    model.add(tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(3,3),
                                            activation='relu',
                                            padding='same',
                                            name='g3_conv1'))
    model.add(tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(3,3),
                                            activation='relu',
                                            padding='same',
                                            name='g3_conv2'))
    model.add(tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(3,3),
                                            activation='relu',
                                            padding='same',
                                            name='g3_conv3'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), 
                                           name='g3_maxpool'))
    #group 4
    model.add(tf.keras.layers.Conv2D(filters=128,
                                            kernel_size=(3,3),
                                            activation='relu',
                                            padding='same',
                                            name='g4_conv1'))
    model.add(tf.keras.layers.Conv2D(filters=128,
                                            kernel_size=(3,3),
                                            activation='relu',
                                            padding='same',
                                            name='g4_conv2'))
    model.add(tf.keras.layers.Conv2D(filters=128,
                                            kernel_size=(3,3),
                                            activation='relu',
                                            padding='same',
                                            name='g4_conv3'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                           name='g4_maxpool'))
    #group 5
    model.add(tf.keras.layers.Conv2D(filters=128,
                                            kernel_size=(3,3),
                                            activation='relu',
                                            padding='same',
                                            name='g5_conv1'))
    model.add(tf.keras.layers.Conv2D(filters=128,
                                            kernel_size=(3,3),
                                            activation='relu',
                                            padding='same',
                                            name='g5_conv2'))
    model.add(tf.keras.layers.Conv2D(filters=128,
                                            kernel_size=(3,3),
                                            activation='relu',
                                            padding='same',
                                            name='g5_conv3'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                           name='g5_maxpool'))
    
    #classifier
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
    return model
    
'''    
class VGG16(tf.keras.Model):
    
    def __init__(self,input_shape=(150, 150, 3), n_classes=6):
        super(VGG16, self).__init__()
        
        #conv group1
        self.g1_conv1 = tf.keras.layers.Conv2D(filters=16,
                                            kernel_size=(3,3),
                                            input_shape=input_shape,
                                            activation='relu',
                                            padding='same',
                                            name='g1_conv1')
        
        self.g1_conv2 = tf.keras.layers.Conv2D(filters=16,
                                            kernel_size=(3,3),
                                            activation='relu',
                                            padding='same',
                                            name='g1_conv2')
        
        self.g1_maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='g1_maxpool')
        
        #conv group2
        self.g2_conv1 = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=(3,3),
                                            activation='relu',
                                            padding='same',
                                            name='g2_conv1')
        
        self.g2_conv2 = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=(3,3),
                                            activation='relu',
                                            padding='same',
                                            name='g2_conv2')
        self.g2_maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='g2_maxpool')
        
        #conv group3
        self.g3_conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(3,3),
                                            activation='relu',
                                            padding='same',
                                            name='g3_conv1')
        
        self.g3_conv2 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(3,3),
                                            activation='relu',
                                            padding='same',
                                            name='g3_conv2')
        self.g3_conv3 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(3,3),
                                            activation='relu',
                                            padding='same',
                                            name='g3_conv3')
        self.g3_maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='g3_maxpool')
        
        #conv group4
        self.g4_conv1 = tf.keras.layers.Conv2D(filters=128,
                                            kernel_size=(3,3),
                                            activation='relu',
                                            padding='same',
                                            name='g4_conv1')
        
        self.g4_conv2 = tf.keras.layers.Conv2D(filters=128,
                                            kernel_size=(3,3),
                                            activation='relu',
                                            padding='same',
                                            name='g4_conv2')
        self.g4_conv3 = tf.keras.layers.Conv2D(filters=128,
                                            kernel_size=(3,3),
                                            activation='relu',
                                            padding='same',
                                            name='g4_conv3')
        self.g4_maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='g4_maxpool')
        
        #conv group5
        self.g5_conv1 = tf.keras.layers.Conv2D(filters=128,
                                            kernel_size=(3,3),
                                            activation='relu',
                                            padding='same',
                                            name='g5_conv1')
        
        self.g5_conv2 = tf.keras.layers.Conv2D(filters=128,
                                            kernel_size=(3,3),
                                            activation='relu',
                                            padding='same',
                                            name='g5_conv2')
        self.g5_conv3 = tf.keras.layers.Conv2D(filters=128,
                                            kernel_size=(3,3),
                                            activation='relu',
                                            padding='same',
                                            name='g5_conv3')
        self.g5_maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='g5_maxpool')
        
        #fc层实现分类
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.fc2 = tf.keras.layers.Dense(256, activation='relu')
        self.out = tf.keras.layers.Dense(n_classes, activation='softmax')
        
    def call(self, inputs):
        x1 = self.g1_maxpool(self.g1_conv2(self.g1_conv1(inputs)))
        x2 = self.g2_maxpool(self.g2_conv2(self.g2_conv1(x1)))
        x3 = self.g3_maxpool(self.g3_conv3(self.g3_conv2(self.g3_conv1(x2))))
        x4 = self.g4_maxpool(self.g4_conv3(self.g4_conv2(self.g4_conv1(x3))))
        x5 = self.g5_maxpool(self.g5_conv3(self.g5_conv2(self.g5_conv1(x4))))
       
        x6 = self.flatten(x5)
        x7 = self.fc1(x6)
        x8 = self.fc2(x7)
        return self.out(x8)'''