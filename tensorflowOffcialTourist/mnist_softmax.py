import tensorflow as tf
import numpy as np
import tensorboard as tb
# import input_data
from tensorflow.examples.tutorials.mnist import input_data


# input mnist trainning data
mnist = input_data.read_data_sets('../MNIST_data/', one_hot=True)
# define regeression modle
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784],name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10],name='y-input')
b = tf.Variable(tf.zeros([10]))
W = tf.Variable(tf.zeros([784, 10]))
tf.summary.histogram('W',W)
with tf.name_scope('model'):
    y = tf.matmul(x, W) + b
# define cost function  and optimizer
# cost funtion with softmax with cross entropy
with tf.name_scope('cost_fucntion'):
    cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        logits=y, labels=y_))
    cross_entropy1 = -tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)))
    tf.summary.scalar('cost',cross_entropy)
# optimizer with SGD algorythom
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.003).minimize(cross_entropy1)
# sess
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

merge_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter('../tmp/logs/softmax/0')
writer.add_graph(sess.graph)
# evaluate accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# train
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
    if i%10==0:
        s=sess.run(merge_summary, feed_dict={x:batch_xs,y_:batch_ys})
        writer.add_summary(s,i)
    if i%100 ==0:
        print(
            sess.run(
                accuracy,
                feed_dict={
                    x: mnist.test.images,
                    y_: mnist.test.labels}))
