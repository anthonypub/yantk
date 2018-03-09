import sys
import net_pb2
import weights_pb2
from google.protobuf import text_format

arglen = len(sys.argv)

if arglen != 3:
    print("Usage: compare_weight_files.py REF TEST");
    exit()

ref_weights = weights_pb2.TrainingWeights()
f = open(sys.argv[1], 'r')
text_format.Parse(f.read(), ref_weights)
f.close()


test_weights = weights_pb2.TrainingWeights()
f = open(sys.argv[1], 'r')
text_format.Parse(f.read(), test_weights)
f.close()

def squared_error(w1, w2):
    return (w1 - w2) * (w1 - w2)

def total_squared_error(msg1, msg2):
    if msg1.iteration != msg2.iteration:
        raise('iteration mismatch')
    w1 = msg1.weights
    w2 = msg2.weights
    return squared_error(w1.w_h_00, w2.w_h_00) + \
            squared_error(w1.w_h_01, w2.w_h_01) + \
            squared_error(w1.w_h_10, w2.w_h_10) + \
            squared_error(w1.w_h_11, w2.w_h_11) + \
            squared_error(w1.b_0, w2.b_0) + \
            squared_error(w1.b_1, w2.b_1) + \
            squared_error(w1.w_o_00, w2.w_o_00) + \
            squared_error(w1.w_o_01, w2.w_o_01) + \
            squared_error(w1.w_o_10, w2.w_o_10) + \
            squared_error(w1.w_o_11, w2.w_o_11)
err = 0.0

for i in range(len(ref_weights.iteration_weights)):
    err += total_squared_error(ref_weights.iteration_weights[i], test_weights.iteration_weights[i])

print(err)



