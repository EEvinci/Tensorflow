# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:15:30 2022

@author: wjcy19870122
"""

#所有的超参数都配置在这里，只需改动这里即可
cfg = {'input_shape':[150, 150, 3],#HWC
       'train_dataroot':'dataset/seg_train',
       'test_dataroot':'dataset/seg_test',
       'valid_rate': 0.1,
       'epochs': 20,
       'batch_size': 120,
       'lr': 0.001,
       'decay_rate': 0.95,
       'pretrained':True,
       'save_root': './checkpoints'
       }