import input_data
import tensorflow as tf
import tensorboard
# import mnist data
mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)

# define regression model
x = tf.placeholder(tf.float32, [None, 784], name='x')
y = tf.placeholder(tf.float32, [None, 10], name='labels')

W = tf.Variable(tf.zeros([784, 10]), name='W')
b = tf.Variable(tf.zeros([10]), name='b')
w_h = tf.summary.histogram('weight', W)
b_h = tf.summary.histogram('biases', b)

with tf.name_scope('Wx_b') as scope:
    model = tf.nn.softmax(tf.matmul(x, W) + b)
    tf.summary.histogram('model', model)

# define loss function and optimizer
with tf.name_scope('cost_function') as scope:
    cost_function = -tf.reduce_sum(y * tf.log(model))
    tf.summary.scalar('cost_function', cost_function)

# use SGD algorythom
with tf.name_scope('train') as scope:
    train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost_function)

# init variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

merged_summary_op = tf.summary.merge_all()
writer = tf.summary.FileWriter('../tmp/logs/softmax/', sess.graph)

writer.add_graph(sess.graph)
for _ in range(1000):
    x_batch, y_batch = mnist.train.next_batch(100)
    if _ % 10 == 0:
        s = sess.run(merged_summary_op, feed_dict={x: x_batch, y: y_batch})
        writer.add_summary(s, _)
    sess.run(train_op, feed_dict={x: x_batch, y: y_batch})

correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print (
    sess.run(
        accuracy,
        feed_dict={
            x: mnist.test.images,
            y: mnist.test.labels}))
