//This is a totally unrolled net with no matrices.
//
//Problem is XOR.
//Two inputs, two hidden units, two output units, no biases (yet)

#include <iostream>
#include <cmath>
#include <vector>
#include <functional>

using namespace std;
using namespace std::placeholders; //for bind

class TestNet
{

    //nomenclature:
    //x* = inputs
    //w_h_ji = input -> hidden weights from input i to hidden j 
    //w_o_ji = hidden -> output weights from hidden i to output j
    //net_h_i = hidden net value for unit i
    //net_o_i = output net value for unit i
    //h_i = hidden unit i
    //o_i = output unit i
    public:
        //inputs
        float x0, x1;
        //truth
        float y0, y1;
        //weights from input -> hidden
        float w_h_00, w_h_01, w_h_10, w_h_11, net_h_0, net_h_1, h_0, h_1;
        //weights from hidden -> out
        float w_o_00, w_o_01, w_o_10, w_o_11, net_o_0, net_o_1, o_0, o_1;
        //error for output units
        float d_o_0, d_o_1;
        //error for hidden units
        float d_h_0, d_h_1;
        //float rate = 0.1;

        std::function<float(float)> act_fn;
        std::function<float(float)> act_deriv_fn;
        std::function<float(float, float)> cost_fn;
        std::function<float(float, float)> cost_deriv_fn;

        float sigmoid(float f)
        {
            return 1.0f / (1.0f + exp(-f));
        }

        float sigderiv(float f)
        {
            return f * (1.0 - f);
        }

        float my_tanh(float f)
        {
            return tanh(f);
        }
        
        float tanh_deriv(float f)
        {
            return 1 - (f * f);
        }

        void set_act(const std::string& act)
        {
            if(act == "sigmoid")
            {
                act_fn = bind(TestNet::sigmoid, this, std::placeholders::_1);
                act_deriv_fn = bind(TestNet::sigderiv, this, _1);
            }
            else if(act == "tanh")
            {
                act_fn = bind(TestNet::my_tanh, this, _1);
                act_deriv_fn = bind(TestNet::tanh_deriv, this, _1);
            }
            else
            {
                throw std::runtime_error("bad activation fn");
            }

        }

        float errderiv(float t, float o)
        {
            return t-o;
        }

        float squared_error(float t, float o)
        {
            return (t - o) * (t - o);
        }

        void InitializeWeights()
        {
            /*
            w_h_00 = 2.4096e-02;
            w_h_01 = -2.2595e-02;
            w_h_10 = -1.8613e-02;
            w_h_11 = 9.4031e-03;
            w_o_00 = -7.0273e-04;
            w_o_01 = -1.7906e-02;
            w_o_10 = -1.4896e-02;
            w_o_11 = 5.9187e-04;
            */
            w_h_00 = 2.4096e-01;
            w_h_01 = -2.2595e-01;
            w_h_10 = -1.8613e-01;
            w_h_11 = 9.4031e-01;
            w_o_00 = -7.0273e-01;
            w_o_01 = -1.7906e-01;
            w_o_10 = -1.4896e-01;
            w_o_11 = 5.9187e-01;
        }

        //Returns error for an example
        float iterate(float x0, float x1, float y0, float y1, bool dump, float rate)
        {
            //forward
            net_h_0 = x0 * w_h_00;
            net_h_0 += x1 * w_h_01; 
            net_h_1 = x0 * w_h_10;
            net_h_1 += x1 * w_h_11; 
            h_0 = act_fn(net_h_0);
            h_1 = act_fn(net_h_1);
            net_o_0 = h_0 * w_o_00;
            net_o_0 += h_1 * w_o_01;
            net_o_1 = h_0 * w_o_10;
            net_o_1 += h_1 * w_o_11;
            o_0 = act_fn(net_o_0);
            o_1 = act_fn(net_o_1);

            //backward
            d_o_0 = errderiv(y0, o_0) * act_deriv_fn(o_0);
            d_o_1 = errderiv(y1, o_1) * act_deriv_fn(o_1);
            float downstream_error_h0 = w_h_00 * d_o_0;
            downstream_error_h0 += w_h_01 * d_o_1;
            d_h_0 = act_deriv_fn(h_0) * downstream_error_h0;
            float downstream_error_h1 = w_h_10 * d_o_0;
            downstream_error_h1 += w_h_11 * d_o_1;
            d_h_1 = act_deriv_fn(h_1) * downstream_error_h1;

            if(dump)
            {
                cout << "truth y0: " << y0 << ", pred y0 " << o_0 << ", diff " << abs(y0 - o_0) << endl;
                cout << "truth y1: " << y1 << ", pred y1 " << o_1 << ", diff " << abs(y1 - o_1) << endl;
            }

            //Now do the updates.
            w_o_00 += rate * d_o_0 * h_0;
            w_o_01 += rate * d_o_0 * h_1;
            w_o_10 += rate * d_o_1 * h_0;
            w_o_11 += rate * d_o_1 * h_1;

            w_h_00 += rate * d_h_0 * x0;
            w_h_01 += rate * d_h_0 * x1;
            w_h_10 += rate * d_h_1 * x0;
            w_h_11 += rate * d_h_1 * x1;

            //error
            float err = squared_error(y0, o_0) + squared_error(y1, o_1);
            return err;
        }

        //Does a forward + backward iteration for all the training examples,
        //returns total error.
        float run_all_examples(bool dump, float rate)
        {
            float err = iterate(0.0, 0.0, 0.0, 1.0, dump, rate);
            err += iterate(0.0, 1.0, 1.0, 0.0, dump, rate);
            err += iterate(1.0, 0.0, 1.0, 0.0, dump, rate);
            err += iterate(1.0, 1.0, 0.0, 1.0, dump, rate);
            return err / 2.0;
        }

};

int main(int argc, char** argv)
{
    TestNet tn;
    tn.InitializeWeights();
    int iters = 1000;
    string act = "sigmoid";
    float rate = 0.05;
    if(argc > 1) 
    {
        iters = atoi(argv[1]);
    }
    if(argc > 2)
    {
        act = argv[2];
    }
    if(argc > 3)
    {
      rate = atof(argv[3]); 
    }

    tn.set_act(act);
    for(int i=0; i < iters; ++i)
    {
        float err = tn.run_all_examples(i % 10 == 0, rate);
        cout << "Err for iter " << i << ": " << err << endl;
    }
}