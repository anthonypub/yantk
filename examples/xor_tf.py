import tensorflow as tf
import sys
import net_pb2
import weights_pb2
from google.protobuf import text_format


arglen = len(sys.argv)

if arglen == 2:
    print('single arg, assuming prototxt msg')
    net_desc_msg = net_pb2.NetDesc()
    f = open(sys.argv[1], 'r')
    text_format.Parse(f.read(), net_desc_msg)
    f.close()
    iters = net_desc_msg.num_iterations
    lr = net_desc_msg.learning_rate
    do_batch = net_desc_msg.batch
    nonlin_name = 'sigmoid'

else:
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
    if arglen > 4 and sys.argv[4] == 'batch':
        do_batch = True;
    else:
        do_batch = False;

net_desc_msg = net_pb2.NetDesc()
net_desc_msg.num_iterations = iters
net_desc_msg.learning_rate = lr
net_desc_msg.batch = do_batch
f = open("desc.out", "w")
f.write(text_format.MessageToString(net_desc_msg))
f.close()


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
if do_batch:
    X = tf.constant([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], name="X")
    y = tf.constant([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], name="X")
    #X = tf.constant([[0.0, 0.0], [0.0, 1.0]], name="X")
    #y = tf.constant([[0.0, 1.0], [1.0, 0.0]], name="y")

    X_feed = X
    Y_feed = y

else:
    X = tf.placeholder(shape=[1, 2], dtype=tf.float32, name="X")
    y = tf.placeholder(shape=[1, 2], dtype=tf.float32, name="y")
    X_feed = [[[0.0, 0.0]], [[0.0, 1.0]], [[1.0, 0.0]], [[1.0, 1.0]] ]
    Y_feed = [[[0.0, 1.0]], [[1.0, 0.0]], [[1.0, 0.0]], [[0.0, 1.0]] ]
#X_feed = [[[0.0, 0.0]] ]
#Y_feed = [[[0.0, 1.0]] ]

#y = tf.constant([[0.0, 1.0], [1.0, 0.0]], name="y")

#Weights from x -> h
W1 = tf.Variable([[2.4096e-01, -1.8613e-01 ], [-2.2595e-01, 9.4031e-01]], name="W1")
#Weights from h -> o
W2 = tf.Variable([[-7.0273e-01, -1.4896e-01], [-1.7906e-01, 5.9187e-01]], name="W2")
#Bias from h -> o
B = tf.Variable([[0.0, 0.0]])

net_h = tf.matmul(X, W1)
#act_h = tf.sigmoid(net_h)
act_h = nonlin(net_h)
net_o = tf.matmul(act_h, W2) + B
#act_o = tf.sigmoid(net_o)
act_o = nonlin(net_o)
err = y - act_o
squared_err = err ** 2.0

cost = tf.reduce_sum(squared_err / 2.0)
grad_o = tf.gradients(xs=act_o, ys=cost)
grad_o_net = tf.gradients(xs=net_o, ys=act_o)
grad_b = tf.gradients(xs=B, ys=cost)
grad_h = tf.gradients(xs=act_h, ys=cost)
grad_h_net = tf.gradients(xs=net_h, ys=act_h)
grad_w1 = tf.gradients(xs=W1, ys=cost)
grad_w2 = tf.gradients(xs=W2, ys=cost)

train_step = tf.train.GradientDescentOptimizer(lr).minimize(cost)

print('xo: ', X_feed[0])

report_freq = 10000

training_weights_msg = weights_pb2.TrainingWeights()

with tf.Session() as sess:
    sess.run(W1.initializer)
    sess.run(W2.initializer)
    sess.run(B.initializer)
    for i in range(iters):
        iter_weights_msg = training_weights_msg.iteration_weights.add()
        iter_weights_msg.iteration = i
        weights_msg = iter_weights_msg.weights
        weights_msg.w_h_00 = 0.0
        weights_msg.w_h_01 = 0.0
        weights_msg.w_h_10 = 0.0
        weights_msg.w_h_11 = 0.0
        weights_msg.b_0 = 0.0
        weights_msg.b_1 = 0.0
        weights_msg.w_o_00 = 0.0
        weights_msg.w_o_01 = 0.0
        weights_msg.w_o_10 = 0.0
        weights_msg.w_o_11 = 0.0
        

        report = (i == iters-1) or i % report_freq == 0
        total_cost=0.0
        j = 0
        if do_batch:
            samples=1
        else:
            samples=len(X_feed)
        while j < samples:
            if do_batch:
                dict = None 
            else:
                dict = {X: X_feed[j], y: Y_feed[j]}
            if report:
                print('start pre-run weights')
                print('w1: ', sess.run(W1))
                print('b: ', sess.run(B))
                print('w2: ', sess.run(W2))
                print('end pre-run weights')
            curr_cost = sess.run(cost, feed_dict=dict)
            total_cost += curr_cost
            if report:
                print('cost at ', i, ":", curr_cost )
            if report:
                print('h: ', sess.run(act_h, feed_dict=dict))
                print('o: ', sess.run(act_o, feed_dict=dict))
                print('err: ', sess.run(err, feed_dict=dict))
                print('sq_err: ', sess.run(squared_err, feed_dict=dict))
                print('grad_o: ', sess.run(grad_o, feed_dict=dict))
                print('grad_o_net: ', sess.run(grad_o_net, feed_dict=dict))
                print('grad_w2: ', sess.run(grad_w2, feed_dict=dict))
                print('grad_b: ', sess.run(grad_b, feed_dict=dict))
                print('grad_h: ', sess.run(grad_h, feed_dict=dict))
                print('grad_w1: ', sess.run(grad_w1, feed_dict=dict))
            sess.run(train_step, feed_dict=dict)
            j = j + 1 
        if report:
            print('total cost at : ',i, ':', total_cost)


    f = open("weights.out", "w")
    f.write(text_format.MessageToString(training_weights_msg))
    f.close()

