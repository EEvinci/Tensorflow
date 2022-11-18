# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 19:22:38 2022

@author: wjcy19870122
"""

import imageio
import os
import os.path as osp
import tensorflow as tf
from vgg16 import VGG16
from cfgs import cfg
from data_loader import DataLoader
from transforms import TransformCompose, Resize
from utils import save_features
    
h, w = cfg['input_shape'][0:2]
test_data_loader = DataLoader(cfg['test_dataroot'],
                              TransformCompose([Resize(h, w)]))

model = VGG16(input_shape = tuple(cfg['input_shape']),
              n_classes=test_data_loader.get_num_categories())

#类定义的模型需要调用build进行实例化，告诉模型输入大小
model.build(input_shape= tuple([cfg['batch_size']] + cfg['input_shape']))

checkpoint_file = osp.join(cfg['save_root'],'weights_best.h5')
if osp.exists(checkpoint_file):
    print('loading pretrained weights from ', checkpoint_file)
    model.load_weights(checkpoint_file, by_name=True)
            
#打印每一层网络的名称
for layer in model.layers:
    print(layer.name)
    
#指定需要输出层的名称，并获取对应的output
output_names = ['g1_conv2', 'g2_conv2', 'g3_conv3', 'g4_conv3', 'g5_conv3']
outputs = []
for name in output_names:
    outputs.append(model.get_layer(name).output)
    
#重新实例化一个model，指定对应的输入和输出
sub_model = tf.keras.Model(inputs = model.input, outputs = outputs)
xs, ys = test_data_loader.next_batch(1)
x1, x2, x3, x4, x5 = sub_model(xs)
print(x1.shape, x2.shape, x3.shape, x4.shape, x5.shape)

save_root = 'features_visualization'
if not osp.exists(save_root): os.mkdir(save_root)

save_features(output_names, [x1.numpy(), x2.numpy(), x3.numpy(), x4.numpy(), x5.numpy()], save_root)
imageio.imsave(osp.join(save_root, 'im.png'), xs[0,...])





