import tensorboard as tb
import tensorflow as tf
import numpy as np
import input_data
# import mnist trainning data
mnist = input_data.read_data_sets('../MNIST_data/',one_hot=True)

# define input and output
x = tf.placeholder(tf.float32, [None, 784], name="x")
# weight and biases
W = tf.Variable(tf.zeros([784, 10]), name='W')
b = tf.Variable(tf.zeros([10]), name='b')
w_h = tf.summary.histogram('weight', W)
b_h = tf.summary.histogram('biases', b)

# y = tf.nn.softmax(tf.matmul(x, W) + b)
y = tf.matmul(x,W)+b
y_ = tf.placeholder(tf.float32, [None, 10])

# define coss_functin

# cost_function = -tf.reduce_sum(y_ * tf.log(y)+(1-y_)*tf.log(1-y))
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_))
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost_function)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(1000):
    x_batch, y_batch = mnist.train.next_batch(100)
    sess.run(train_op, feed_dict={x: x_batch, y_: y_batch})

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print (sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))
