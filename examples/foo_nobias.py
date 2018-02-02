from __future__ import print_function

import theano
import theano.tensor as T
import numpy as np
import time

X = theano.shared(value=np.asarray([[0, 1], [1, 0], [0, 0], [1, 1]]), name='X')
y = theano.shared(value=np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]]), name='y')
#rng = np.random.RandomState(1234)
LEARNING_RATE = 0.1

#def layer(*shape):
	#mag = 4. * np.sqrt(6. / sum(shape))
	#return theano.shared(value=np.asarray(rng.uniform(low=-mag, high=mag,
		#size=shape), dtype=theano.config.floatX), name='W', borrow=True, strict=False)

#W1 = layer(2, 2)
W1 = theano.shared(value=np.asarray([[2.4096e-01, -2.2595e-01], [-1.8613e-01, 9.4031e-01]]), name="W")
W2 = theano.shared(value=np.asarray([[-7.0273e-01, -1.7906e-01], [-1.4896e-01, 5.9187e-01]]), name="W")

#output = T.nnet.sigmoid(T.dot(T.nnet.relu(T.dot(X, W1) ), W2) )
output = T.nnet.sigmoid(T.dot(T.nnet.sigmoid(T.dot(X, W1) ), W2) )
cost = T.sum((y - output) ** 2)
updates = [(W1, W1 - LEARNING_RATE * T.grad(cost, W1)),
		(W2, W2 - LEARNING_RATE * T.grad(cost, W2))]

train = theano.function(inputs=[], outputs=[], updates=updates)
test = theano.function(inputs=[], outputs=cost)

print('Error before:', test())
start = time.time()
for i in range(10):
	train()
        print(output)
end = time.time()
print('Error after:', test())
print('Time (s):', end - start)
