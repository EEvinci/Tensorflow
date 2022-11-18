import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print('tensorflow verions:', tf.__version__)
 

#准备数据集：train、validate、test
minist = tf.keras.datasets.mnist
(train_ims, train_labels), (test_ims, test_labels) = minist.load_data()

total_num, H, W = train_ims.shape[0:3]
valid_split = 0.2#验证集比例20%
train_num = int(total_num*(1-valid_split))

train_x = train_ims[:train_num]
train_y = train_labels[:train_num]

valid_x = train_ims[train_num:]
valid_y = train_labels[train_num:]

test_x = test_ims
test_y = test_labels

print('train/valid/test:', train_y.shape[0], valid_y.shape[0], test_y.shape[0])

#将数据reshape为（total_num, 768），因为升级网络的输入是特征向量
train_x = train_x.reshape(-1, H*W)
valid_x = valid_x.reshape(-1, H*W)
test_x = test_x.reshape(-1, H*W)

#将数据归一化处理
train_x = tf.cast(train_x/255.0, tf.float32)
valid_x = tf.cast(valid_x/255.0, tf.float32)
test_x = tf.cast(test_x/255.0, tf.float32)

#将标签进行独热编码
class_num = len(np.unique(train_labels))#10
print('number of classes:', class_num)
train_y = tf.one_hot(train_y, class_num)
valid_y = tf.one_hot(valid_y, class_num)
test_y = tf.one_hot(test_y, class_num)

#构建神经网络模型
def model(x, w, b):
    pred = tf.matmul(x, w) + b
    return tf.nn.softmax(pred)


#定义损失函数
def loss(x, y, w, b):
    pred = model(x, w, b)
    loss_ = tf.keras.losses.categorical_crossentropy(y_true=y, y_pred=pred)
    return tf.reduce_mean(loss_)

#计算梯度
def grad(x, y, w, b):
    with tf.GradientTape() as tape:
        loss_ = loss(x, y, w, b)
    return tape.gradient(loss_, [w, b])

#定义准确率
def accuracy(x, y, w, b):
    pred = model(x, w, b)
    corrections = tf.equal(tf.argmax(pred, axis=-1), tf.argmax(y, axis=-1))
    return tf.reduce_mean(tf.cast(corrections, tf.float32))

epochs = 10
batch_size = 60
lr = 0.001

optimizer = tf.keras.optimizers.Adam(learning_rate = lr)

steps = train_num//batch_size

train_losses = []
valid_losses = []
train_accuracys = []
valid_accuracys = []

#定义待优化的模型参数
W = tf.Variable(tf.random.normal([H*W, class_num], mean=0.0, stddev=1.0, dtype=tf.float32))
B = tf.Variable(tf.zeros([class_num], dtype=tf.float32))

for epoch in range(epochs):
    for step in range(steps):
        xs = train_x[step*batch_size:(step+1)*batch_size]
        ys = train_y[step*batch_size:(step+1)*batch_size]

        grads = grad(xs, ys, W, B)
        optimizer.apply_gradients(zip(grads, [W, B]))

    train_loss = loss(train_x, train_y, W, B).numpy()
    valid_loss = loss(valid_x, valid_y, W, B).numpy()
    train_accuracy = accuracy(train_x, train_y, W, B).numpy()
    valid_accuracy = accuracy(valid_x, valid_y, W, B).numpy()

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_accuracys.append(train_accuracy)
    valid_accuracys.append(valid_accuracy)

    print('epoch={:3d}, train_loss={:.4f}, train_acc={:.4f}, val_loss={:.4f}, val_acc={:.4f}'.format(\
        epoch+1, train_loss, train_accuracy, valid_loss, valid_accuracy))

test_accuracy = accuracy(test_x, test_y, W, B).numpy()
print('test accuracy:', test_accuracy)

#显示训练过程中的损失和准确率
plt.subplot(1,2,1)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(train_losses, 'blue', label='train loss')
plt.plot(valid_losses, 'red', label='valid loss')
plt.legend(loc=1)

plt.subplot(1,2,2)
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.plot(train_accuracys, 'blue', label='train accuracy')
plt.plot(valid_accuracys, 'red', label='valid accuracy')
plt.legend(loc=1)

plt.show()