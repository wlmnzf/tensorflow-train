# load MNIST data
# coding=utf-8
import os
import input_data
from mnist_demo import * 
mnist = input_data.read_data_sets("data/", one_hot=True)



# start tensorflow interactiveSession
import tensorflow as tf
sess = tf.InteractiveSession()

# weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    # print(tf.Variable(initial).eval())
    return tf.Variable(initial)

# http://www-rohan.sdsu.edu/doc/matlab/techdoc/ref/convn.html
# convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def print_matrix(board):
    # board.Format(_T("x=%.18e,y=%.18e"), point.x, point.y);
    # board=array(board,dtype="float16")
    # savetxt('new.txt', board,'%.16g')  
    shapex=shape(board)
    batch=shapex[0]
    x=shapex[1]
    y=shapex[2]
    channel=shapex[3]

    for i in range(batch):
        tmp=zeros(shape=(x,y))
        for j in range(x):
          for k in range(y):
            tmp[j][k]=board[i][j][k][0]

        for row in tmp:
          print (row)
          print "\n"

# Create the model
# placeholder
x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 10])
# variables
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)

# first convolutinal layer  h_conv1(0,28)
w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])


x_image = tf.reshape(x, [-1, 28, 28, 1])


# 一句话概括：不用simgoid和tanh作为激活函数，而用ReLU作为激活函数的原因是：加速收敛。
# 因为sigmoid和tanh都是饱和(saturating)的。何为饱和？个人理解是把这两者的函数曲线和导数曲线plot出来就知道了：他们的导数都是倒过来的碗状，也就是，越接近目标，对应的导数越小。而ReLu的导数对于大于0的部分恒为1。于是ReLU确实可以在BP的时候能够将梯度很好地传到较前面的网络。
# ReLU（线性纠正函数）取代sigmoid函数去激活神经元

#relu激活函数跟sigmod类似，把数值转为0-1，但是有更多的优点，加速收敛。
# x_image (1, 28, 28, 32)   w_conv1 (5, 5, 1, 32)   b_conv1(32)
#conv  [batch, in_height, in_width, in_channels]    [filter_height, filter_width, in_channels, out_channels]
#
#  Flattens the filter to a 2-D matrix with shape [filter_height * filter_width * in_channels, output_channels].
#  Extracts image patches from the the input tensor to form a virtual tensor of shape [batch, out_height, out_width, filter_height * filter_width * in_channels].
#  For each patch, right-multiplies the filter matrix and the image patch vector.
#


#x_image(1, 28, 28, 1)
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
# h_conv1 (1, 28, 28, 32)
# h_pool1(1, 14, 14, 32)
# ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]

# second convolutional layer
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# h_conv2 (1, 14, 14, 64)
# h_pool2 (1, 7, 7, 64)

# densely connected layer
w_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)



# train and evaluate the model
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdagradOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
accuracy=tf.cast(tf.argmax(y_conv, 1),tf.float32)
sess.run(tf.initialize_all_variables())


# print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels, keep_prob:1.0}))


# for i in range(1):
#     batch = mnist.train.next_batch(50)
#     if i%100 == 0:
#         train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
#         print "step %d, train accuracy %g" %(i, train_accuracy)
#     train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})


# print (accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))
for i in range(1):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    # print (i, train_accuracy)
    print(i)

  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# print(w_conv1)
# ddd=conv2d(x_image, w_conv1) 
dir_name="test_num"
files = os.listdir(dir_name)
cnt=len(files)
for i in range(cnt):
  files[i]=dir_name+"/"+files[i]
  test_images1,test_labels1=GetImage([files[i]])
  mnist.test = input_data.DataSet(test_images1, test_labels1, dtype=tf.float32)
  res=y_conv.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
  # res=h_pool2.eval()
  # res=accuracy.eval({x: mnist.test.images, y_: mnist.test.labels})
  # print("output:",int(res[0]))
  # print_matrix(res)
  print("output:",shape(res))
  break

