from __future__ import print_function

import theano
import theano.tensor as T
import numpy as np
import time

#X = theano.shared(value=np.asarray([[0, 1], [1, 0], [0, 0], [1, 1]]), name='X')
#y = theano.shared(value=np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]]), name='y')
#single example
X = theano.shared(value=np.asarray([[0, 1]]), name='X')
y = theano.shared(value=np.asarray([[1, 0]]), name='y')
#rng = np.random.RandomState(1234)
LEARNING_RATE = 0.1

#def layer(*shape):
	#mag = 4. * np.sqrt(6. / sum(shape))
	#return theano.shared(value=np.asarray(rng.uniform(low=-mag, high=mag,
		#size=shape), dtype=theano.config.floatX), name='W', borrow=True, strict=False)

#W1 = layer(2, 2)
#I had these transposed.
#W1 = theano.shared(value=np.asarray([[2.4096e-01, -2.2595e-01], [-1.8613e-01, 9.4031e-01]]), name="W1")
#W2 = theano.shared(value=np.asarray([[-7.0273e-01, -1.7906e-01], [-1.4896e-01, 5.9187e-01]]), name="W2")

W1 = theano.shared(value=np.asarray([[2.4096e-01, -1.8613e-01 ], [-2.2595e-01, 9.4031e-01]]), name="W1")
W2 = theano.shared(value=np.asarray([[-7.0273e-01, -1.4896e-01], [-1.7906e-01, 5.9187e-01]]), name="W2")

#output = T.nnet.sigmoid(T.dot(T.nnet.relu(T.dot(X, W1) ), W2) )
output = T.nnet.sigmoid(T.dot(T.nnet.sigmoid(T.dot(X, W1) ), W2) )
cost = T.sum(((y - output) ** 2) / 2)
updates = [(W1, W1 - LEARNING_RATE * (T.grad(cost, W1))),
		(W2, W2 - LEARNING_RATE * T.grad(cost, W2))]

train = theano.function(inputs=[], outputs=[], updates=updates)
test = theano.function(inputs=[], outputs=cost)

grad_w1_fn = theano.function(inputs=[], outputs=T.grad(cost, W1))
grad_w2_fn = theano.function(inputs=[], outputs=T.grad(cost, W2));

theano.printing.debugprint(T.grad(cost, W2))
theano.printing.debugprint(grad_w2_fn);

start = time.time()
for i in range(2):
    print("W1=", W1.get_value())
    print("W2=", W2.get_value())
    print('Error at :', i, test())
    train()
    print("grad_w1: ", grad_w1_fn());
    print("grad_w2: ", grad_w2_fn());
end = time.time()
print('Time (s):', end - start)
