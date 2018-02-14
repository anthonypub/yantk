import tensorflow as tf

X = tf.constant([[0.0, 1.0]])
y = tf.constant([[1.0,0.0]])

W1 = tf.Variable([[2.4096e-01, -1.8613e-01 ], [-2.2595e-01, 9.4031e-01]])
W2 = tf.Variable([[-7.0273e-01, -1.4896e-01], [-1.7906e-01, 5.9187e-01]])

net_h = tf.matmul(X, W1)
act_h = tf.sigmoid(net_h)
net_o = tf.matmul(X, W2)
act_o = tf.sigmoid(net_o)

cost = tf.reduce_sum(((y - act_o) ** 2.0) / 2.0)

grad_o = tf.gradients(act_o, cost)

with tf.Session() as sess:
    sess.run(W1.initializer)
    sess.run(W2.initializer)
    print(sess.run(cost))


