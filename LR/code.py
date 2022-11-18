import tensorflow as tf    # 载入Tensorflow
import numpy as np     # 载入numpy
import matplotlib.pyplot as plt # 载入matplotlib
# import sklearn as skl

# print("Tensorflow版本是：",tf.__version__) #显示当前TensorFlow版本

# 直接采用np生成等差数列的方法，生成100个点，每个点的取值在0~1之间
x_data = np.linspace(1, 100, 500)
x_data = x_data / 100

# print(x_data.shape)
np.random.seed(50)    # 设置随机数种子
# y = 2x +1 + 噪声， 其中，噪声的维度与x_data一致
y_data = 3.1234 * x_data + 2.98 + np.random.randn(*x_data.shape) * 0.04

# print(x_data)
# print(y_data)

plt.scatter(x_data, y_data)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Figure: Training Data")
# plt.show()
#画出随机生成数据的散点图
plt.scatter(x_data, y_data)
# plt.show()

# 画出我们想要通过学习得到的目标线性函数 y = 2x +1
plt.plot (x_data, 2.98 + 3.1234 * x_data, 'r', linewidth=3)
plt.show()

# 通过模型执行，将实现前向计算（预测计算）

def model(x, w, b):
    return tf.multiply(x, w) + b

# 构建模型中的变量w，对应线性函数的斜率
w = tf.Variable(np.random.randn(),tf.float32)

# 构建模型中的变量b，对应线性函数的截距
b = tf.Variable(0.0,tf.float32)

# 定义均方差损失函数

def loss(x, y, w, b):
    err = model(x, w, b) - y    #  计算模型预测值和标签值的差异
    squared_err = tf.square(err)    #  求平方，得出方差
    return tf.reduce_mean(squared_err)   # 求均值，得出均方差.

training_epochs, learning_rate = 10, 0.001    # 迭代次数（训练轮数）与学习率（步长）

# 计算样本数据[x,y]在参数[w,b]点上的梯度
def grad(x, y, w, b):
    with tf.GradientTape() as tape:
        loss_ = loss(x, y, w, b)
    return tape.gradient(loss_, [w, b])    # 返回梯度向量


step = 0  # 记录训练步数
loss_list = []  # 用于保存loss值的列表
display_step = 20  # 控制训练过程数据显示的频率，不是超参数

for epoch in range(training_epochs):
    for xs, ys in zip(x_data, y_data):

        loss_ = loss(xs, ys, w, b)  # 计算损失
        loss_list.append(loss_)  # 保存本次损失计算结果

        delta_w, delta_b = grad(xs, ys, w, b)  # 计算该当前[w,b]点的梯度
        change_w = delta_w * learning_rate  # 计算变量w需要调整的量

        change_b = delta_b * learning_rate  # 计算变量b需要调整的量
        w.assign_sub(change_w)  # 变量w值变更为减去chage_w后的值
        b.assign_sub(change_b)  # 变量b值变更为减去chage_b后的值

        step = step + 1  # 训练步数+1
        if step % display_step == 0:  # 显示训练过程信息
            print("Training Epoch:", '%02d' % (epoch + 1), "Step: %03d" % (step), "loss=%.6f" % (loss_))
    plt.plot(x_data, w.numpy() * x_data + b.numpy())  # 完成一轮训练后，画出回归的线条
plt.show()

# print ("w：", w.numpy()) # w的值应该在2附近
# print ("b：", b.numpy()) # b的值应该在1附近

plt.scatter(x_data,y_data,label='Original data')
plt.plot (x_data, x_data * 3.1234 + 2.98, label='Object line',color='g', linewidth=3)
plt.plot (x_data, x_data * w.numpy() + b.numpy(),label='Fitted line',color='r', linewidth=3)
plt.legend(loc=2)# 通过参数loc指定图例位置
plt.show()

x_test = 5.79

predict = model(x_test,w.numpy(),b.numpy())
print("预测值：%f" % predict)

target = 2 * x_test + 1.0
print("目标值：%f" % target)

plt.plot(loss_list)
plt.show()