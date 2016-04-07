# coding=utf-8
import os
import input_data
from mnist_demo import * 
import tensorflow as tf

mnist = input_data.read_data_sets("data/", one_hot=True)


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
    return tf.nn.conv2d(x, W, strides=[1, 3, 3, 1], padding='VALID')
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


sess = tf.InteractiveSession()

x = tf.placeholder("float", [None, 25])
x_image = tf.reshape(x, [-1, 5, 5, 1])

w_conv1=[1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0]
w_conv1= tf.reshape(w_conv1, [3,3, 1, 1])

# w_conv1 = weight_variable([3, 3, 1, 1])

# b_conv1 = bias_variable([1])
b_conv1=[0]

h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
#SOURCE
#1 1 1 0 0
#0 1 1 1 0
#0 0 1 1 1
#0 0 1 1 0
#0 1 1 0 0

#SAME_SOURCE
#外部包一层0


# SAME
#2 2 3 1 1
#1 4 3 4 1
#1 2 4 3 3 
#1 2 3 4 1
#0 2 2 1 1


# VALID
#4 3 4
#2 4 3
#2 3 4


# test_images1,test_labels1=GetImage(["test_num/0_0.png"])
# mnist.test = input_data.DataSet(test_images1, test_labels1, dtype=tf.float32)

sess.run(tf.initialize_all_variables())
tmp=[[1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,1,1,0,0,1,1,0,0]]
tmp=array(tmp)
# print shape(tmp)

res=h_conv1.eval({x:tmp})
# print print_matrix(res)
print res


# dir_name="test_num"
# files = os.listdir(dir_name)
# cnt=len(files)
# for i in range(cnt):
#   files[i]=dir_name+"/"+files[i]
#   test_images1,test_labels1=GetImage([files[i]])
#   mnist.test = input_data.DataSet(test_images1, test_labels1, dtype=tf.float32)
#   res=h_pool1.eval(feed_dict={x: mnist.test.images})
#   # res=h_pool2.eval()
#   # res=accuracy.eval({x: mnist.test.images, y_: mnist.test.labels})
#   # print("output:",int(res[0]))
#   print_matrix(res)
#   print("output:",shape(res))
#   break

