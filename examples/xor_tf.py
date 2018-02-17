import tensorflow as tf
import sys


arglen = len(sys.argv)

if arglen > 1:
    iters = int(sys.argv[1])
else:
    iters = 1
if arglen > 2:
    nonlin_name = sys.argv[2]
else:
    nonlin_name = 'sigmoid'
if arglen > 3:
    lr = float(sys.argv[3])
else:
    lr = 0.1

if nonlin_name == 'sigmoid':
    nonlin = tf.sigmoid
elif nonlin_name == 'tanh':
    nonlin = tf.tanh
elif nonlin_name ==  'relu':
    nonlin = tf.nn.relu
else:
    print('usage: xor_tf.py ITERS NONLIN LEARNRATE')
    raise('unknown nonlinearity')


#X = tf.constant([[0.0, 1.0]], name="X")
X = tf.placeholder(shape=[1, 2], dtype=tf.float32, name="X")
y = tf.placeholder(shape=[1, 2], dtype=tf.float32, name="y")
X_feed = [[[0.0, 0.0]], [[0.0, 1.0]], [[1.0, 0.0]], [[1.0, 1.0]] ]
Y_feed = [[[0.0, 1.0]], [[1.0, 0.0]], [[1.0, 0.0]], [[0.0, 1.0]] ]
#y = tf.constant([[0.0, 1.0], [1.0, 0.0]], name="y")

W1 = tf.Variable([[2.4096e-01, -1.8613e-01 ], [-2.2595e-01, 9.4031e-01]], name="W1")
W2 = tf.Variable([[-7.0273e-01, -1.4896e-01], [-1.7906e-01, 5.9187e-01]], name="W2")

net_h = tf.matmul(X, W1)
#act_h = tf.sigmoid(net_h)
act_h = nonlin(net_h)
net_o = tf.matmul(act_h, W2)
#act_o = tf.sigmoid(net_o)
act_o = nonlin(net_o)
err = y - act_o
squared_err = err ** 2.0

cost = tf.reduce_sum(squared_err / 2.0)
grad_o = tf.gradients(xs=act_o, ys=cost)
grad_o_net = tf.gradients(xs=net_o, ys=act_o)
grad_h = tf.gradients(xs=act_h, ys=cost)
grad_h_net = tf.gradients(xs=net_h, ys=act_h)
grad_w1 = tf.gradients(xs=W1, ys=cost)
grad_w2 = tf.gradients(xs=W2, ys=cost)

train_step = tf.train.GradientDescentOptimizer(lr).minimize(cost)

print('xo: ', X_feed[0])

with tf.Session() as sess:
    sess.run(W1.initializer)
    sess.run(W2.initializer)
    for i in range(iters):
        total_cost=0.0
        for j in range(len(X_feed)):
            dict = {X: X_feed[j], y: Y_feed[j]}
            curr_cost = sess.run(cost, feed_dict=dict)
            total_cost += curr_cost
            if i % 1000 == 0:
                print('cost at ', i, ":", curr_cost )
            if i % 1000 == 0:
                print('h: ', sess.run(act_h, feed_dict=dict))
                print('o: ', sess.run(act_o, feed_dict=dict))
                print('err: ', sess.run(err, feed_dict=dict))
                print('sq_err: ', sess.run(squared_err, feed_dict=dict))
                print('grad_o: ', sess.run(grad_o, feed_dict=dict))
                print('grad_o_net: ', sess.run(grad_o_net, feed_dict=dict))
                print('grad_w2: ', sess.run(grad_w2, feed_dict=dict))
                print('grad_h: ', sess.run(grad_h, feed_dict=dict))
                print('grad_w1: ', sess.run(grad_w1, feed_dict=dict))
            sess.run(train_step, feed_dict=dict)
        print('total cost at : ',i, ':', total_cost)

