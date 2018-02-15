import tensorflow as tf

X = tf.constant([[0.0, 1.0]], name="X")
y = tf.constant([[1.0,0.0]], name="y")

W1 = tf.Variable([[2.4096e-01, -1.8613e-01 ], [-2.2595e-01, 9.4031e-01]], name="W1")
W2 = tf.Variable([[-7.0273e-01, -1.4896e-01], [-1.7906e-01, 5.9187e-01]], name="W2")

net_h = tf.matmul(X, W1)
act_h = tf.sigmoid(net_h)
net_o = tf.matmul(act_h, W2)
act_o = tf.sigmoid(net_o)
err = y - act_o
squared_err = err ** 2.0

cost = tf.reduce_sum(squared_err / 2.0)
grad_o = tf.gradients(xs=act_o, ys=cost)
grad_o_net = tf.gradients(net_o, cost)

with tf.Session() as sess:
    sess.run(W1.initializer)
    sess.run(W2.initializer)
    print('h: ', sess.run(act_h))
    print('o: ', sess.run(act_o))
    print('err: ', sess.run(err))
    print('sq_err: ', sess.run(squared_err))
    print(sess.run(cost))
    print(sess.run(grad_o))
    


