# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:16:24 2022

@author: wjcy19870122
"""
import cv2
import os
import os.path as osp
import glob
import numpy as np
import imageio

class DataLoader(object):
    def __init__(self, data_root, transforms=None):
        '''
        data_root: 数据集根目录; 训练集:dataset\seg_train; 测试集:dataset\seg_test
        transforms:数据预处理和扩增:resize->randomflip->randomrotate
        '''
        assert osp.exists(data_root), 'No data directory found at ' + data_root
       
        self.transforms = transforms #用于数据预测处理和扩增
        self.categories = os.listdir(data_root)#根目录下的文件夹个数对应类别
        
        #检索所有图像文件名,然后构建图像和label对（pairs）
        self.images_labels = []
        for idx, category in enumerate(self.categories):
            cate_ims = glob.glob(osp.join(data_root, category, '*.jpg'))
            cate_ims_labels = [[cate_im, idx] for cate_im in cate_ims]
            self.images_labels.extend(cate_ims_labels)
            
        print('total images:', len(self.images_labels))
        
        self.cur_idx = 0
        #需要打乱样本顺序,才能保障每个batch中包含多个类别图像
        np.random.shuffle(self.images_labels)
        
        
    def next_batch(self, batch_size):
        #用于保存每个batch的图像和标签
        ims = []; labels = []
        
        for i in range(batch_size):
            #如果当前样本的idx超过样本总数,从头开始读取
            idx = (self.cur_idx+i)%len(self.images_labels)
            
            im_file, label = self.images_labels[idx]
            im = imageio.imread(im_file)
                   
            #执行预处理和数据扩增:resize->randomflip->randomrotate
            if self.transforms:
                im = self.transforms(im)
                    
            ims.append(im)
            labels.append(label)
            
        #batch读完后,将当前idx向前移动batch_size,如果越界了则从头开始
        self.cur_idx += batch_size
        if self.cur_idx >= len(self.images_labels):
            self.cur_idx = 0
            
        #将样本构建为numpy数组
        ims = np.array(ims, dtype=np.float32)#[batchsize, 150, 150, 3]
        labels = np.array(labels, dtype=np.int16)#[batchsize]
       
        return ims, labels
    
    def get_num_categories(self):
        return len(self.categories)
    
    def get_num_samples(self):
        return len(self.images_labels)
    
#单元测试,确保DataLoader没错误
if __name__ == '__main__':
    data_loader = DataLoader('dataset/seg_train')
    print(data_loader.get_num_categories())
    steps = data_loader.get_num_samples()//60
    for i in range(steps):
        ims, labels = data_loader.next_batch(60)
        print(ims.shape, labels.shape)