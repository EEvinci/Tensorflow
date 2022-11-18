# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 10:07:21 2022

@author: wjcy19870122
"""

import os
import os.path as osp
import tensorflow as tf
import time
from vgg16 import VGG16
from utils import draw_progress_bar
from cfgs import cfg
from data_loader import DataLoader
from transforms import TransformCompose, Resize, RandomFlip, RandomRotate

def calc_accuracy(ys_gt, ys_pred):
    '''
    ys_gt:[batch_size], ground-truth标签
    ys_pred:[batch_size, 10], 预测的概率分布
    '''    
    ys_pred = tf.argmax(ys_pred, axis=-1)#[batch_size]
    ys_gt = tf.cast(tf.reshape(ys_gt, (-1)), ys_pred.dtype)
    corrections = tf.equal(ys_pred, tf.reshape(ys_gt, (-1)))
    accuracy = tf.reduce_mean(tf.cast(corrections, tf.float32))
    return accuracy

def train_epoch(train_data_loader, model, optimizer, steps, batch_size):
    #记录所有steps的损失和准确率
    losses = []
    accuracys = []
        
    #每个epoch要训练完所有训练集图像       
    for step in range(steps):
        xs, ys = train_data_loader.next_batch(batch_size)
            
        with tf.GradientTape(persistent=False) as tape:
            #将图像输入模型，得到预测概率
            probs= model(xs.astype('float32')) #[batch_size, 6]
           
            #计算预测值和金标准值之间的概率损失函数
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(ys, probs)
            losses.append(loss)
                
            #计算accuracy
            accuracy = calc_accuracy(ys, probs)
            accuracys.append(accuracy)
                
            #根据损失计算所有可训练参数的梯度
            grad = tape.gradient(loss, model.trainable_variables)
                
            #优化所有可训练的模型参数
            optimizer.apply_gradients(grads_and_vars=zip(grad, model.trainable_variables))
            
            #打印进度条
            draw_progress_bar(step+1, steps)
            
    avg_loss = tf.reduce_mean(losses)
    avg_acc = tf.reduce_mean(accuracys)
    return avg_loss, avg_acc

def run_evaluation(test_data_loader, model, batch_size):
    '''
    跑完一个epoch后调用此函数执行验证
    '''
   
    valid_steps = test_data_loader.get_num_samples()//batch_size
        
    #记录所有steps的损失和准确率
    losses = []
    accuracys = []
    for step in range(valid_steps):
        xs, ys = test_data_loader.next_batch(batch_size)
        probs= model(xs.astype('float32')) #[batch_size, 6]
        
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(ys, probs)
        losses.append(loss)
        
        accuracy = calc_accuracy(ys, probs)
        accuracys.append(accuracy)
        
    avg_loss = tf.reduce_mean(losses)
    avg_acc = tf.reduce_mean(accuracys)
    return avg_loss, avg_acc



#主函数，导入数据集->实例化模型->定义训练超参数->定义优化器->迭代训练并保存模型
if __name__ == '__main__':
    #首先创建模型保存的目录    
    save_root = cfg['save_root']
    if not osp.exists(save_root): os.mkdir(save_root)
    
    #======================导入数据集 =====================
    h, w = cfg['input_shape'][0:2]
    train_data_loader = DataLoader(cfg['train_dataroot'],
                                   TransformCompose([Resize(h, w),
                                                     RandomFlip(), 
                                                     RandomRotate()]))
    test_data_loader = DataLoader(cfg['test_dataroot'],
                                  TransformCompose([Resize(h, w)]))
    
    #===================== 实例化CNN模型 ========================
    model = VGG16(input_shape = tuple(cfg['input_shape']),
                  n_classes=train_data_loader.get_num_categories())
    
    #类定义的模型需要调用build进行实例化，告诉模型输入大小
    model.build(input_shape= tuple([cfg['batch_size']] + cfg['input_shape']))
    
    print('模型结构为：')
    model.summary()
    
    if cfg['pretrained'] is True:
        checkpoint_file = osp.join(save_root,'weights_best.h5')
        if osp.exists(checkpoint_file):
            print('loading pretrained weights from ', checkpoint_file)
            model.load_weights(checkpoint_file)
    
    
    #===================== 定义训练超参数 =======================   
    steps_per_epoch = train_data_loader.get_num_samples()//cfg['batch_size']
    decay_steps = steps_per_epoch
    lr_schedule =  tf.keras.optimizers.schedules.ExponentialDecay(cfg['lr'], 
                                                                  decay_steps,
                                                                  cfg['decay_rate'],
                                                                  staircase = True)
       
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
       
    
    #=============== 开始迭代训练并保存模型参数=================   
    max_valid_accuracy = 0.0
    for epoch in range(cfg['epochs']):
        print('Training on epoch:', epoch+1)
        
        t1 = time.time()
        agv_train_loss, avg_train_acc = train_epoch(train_data_loader, 
                                                    model, 
                                                    optimizer, 
                                                    steps_per_epoch,
                                                    cfg['batch_size'])        
        t2 = time.time()
      
        #完成了一个epoch之后:1）执行validation；2)打印训练信息；3）根据validation结果考虑是否保存当前模型参数       
        agv_valid_loss, agv_valid_acc = run_evaluation(test_data_loader, 
                                                       model,
                                                       cfg['batch_size'])        
       
        current_lr = tf.keras.backend.eval(optimizer._decayed_lr('float32'))
        print ('\n train loss: %f; train acc: %f; valid loss: %f; valid acc: %f; Lr: %f; used time (s): %f' % \
                      (agv_train_loss, avg_train_acc, agv_valid_loss, agv_valid_acc, current_lr, t2-t1)) 
            
        if max_valid_accuracy < agv_valid_acc:
            max_valid_accuracy = agv_valid_acc  
    
            #只保存验证机准确率最优的模型参数
            checkpoint_file = osp.join(save_root,'weights_best.h5')
            print('saving weights to ', checkpoint_file)
            model.save_weights(checkpoint_file)

    
    
                  
        
    