# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 16:39:05 2022

@author: wjcy19870122
"""
import cv2
import numpy as np

class TransformCompose(object):
    def __init__(self, transforms):
        '''
        transforms： 数据预处理和扩增操作的列表
        '''
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image= t(image)
        
        return image

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
    
    
class Resize(object):
    def __init__(self, target_h = 150, target_w = 150):
        self.target_h = target_h
        self.target_w = target_w
        
    def __call__(self, im):
        return cv2.resize(im, (self.target_w, self.target_h))
    
class RandomFlip(object):
    def __call__(self, im):
        random_flip = np.random.randint(0, 2)
        if random_flip == 0:
            return im
        #将图像进行上下翻转
        elif random_flip == 1:
            im = np.flip(im, axis=0)           
            return im
        else:#将图像进行左右翻转
            im = np.flip(im, axis=1)           
            return im        
    
class RandomRotate(object):
    def __call__(self, im):
        #随机旋转0， 90， 180，270度
        random_angle = np.random.randint(0, 3) * 90
        im = np.rot90(im, random_angle//90)
        return im
        