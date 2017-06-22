import tensorflow as tf
import numpy as np

# prepared train data
M = 100
N = 2
w_data = np.mat([[1.0, 3.0]]).T
b_data = 10
x_data = np.random.randn(M, N).astype(np.float32)
y_data = np.mat(x_data) * w_data + 10 + np.random.randn(M, 1) * 0.33

# define modle and graph
w = tf.Variable(tf.random_uniform([N, 1], -1, 1))
w = tf.Variable([[0.], [0.]], tf.float32)
b = tf.Variable(tf.random_uniform([1], -1, 1))

y = tf.matmul(x_data, w) + b
loss = tf.reduce_mean(tf.square(y - y_data))

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        sess.run(train_op)
        if i % 10 == 0:
            print (sess.run(w).T, sess.run(b))
