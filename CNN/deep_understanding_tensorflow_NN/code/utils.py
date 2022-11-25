# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:14:09 2022

@author: wjcy19870122
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def draw_progress_bar(cur, total, bar_len=50, maker='='):
    '''
    辅助性函数，用于训练迭代过程中打印进度条
    '''
    
    cur_len = int(cur/total*bar_len)
    sys.stdout.write('\r')
    sys.stdout.write("[{:<{}}] {}/{}".format(maker * cur_len, bar_len, cur, total))
    sys.stdout.flush()
    
def save_features(layer_names, layer_features, save_path):

    for layer_name, layer_feature in zip(layer_names, layer_features):
       
        #特征图的形状(1,size,size,n_features)
        size = layer_feature.shape[1]
        n_cols = layer_feature.shape[3]//8
        images_per_row = layer_feature.shape[3]//n_cols
       
        display_grid = np.zeros((size * n_cols,images_per_row * size))

        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_feature[0,:,:,col * images_per_row + row]
                #归一化
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 128
                channel_image += 128
                channel_image = np.clip(channel_image,0,255).astype(np.uint8)
                display_grid[col * size : (col + 1) * size,row * size: (row + 1) * size] = channel_image
        
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid,aspect='auto',cmap='viridis')
        plt.savefig(os.path.join(save_path, layer_name+'.png'))
