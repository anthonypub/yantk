//This is a totally unrolled net with no matrices.
//
//Problem is XOR.
//Two inputs, two hidden units, two output units, no biases (yet)

#include <iostream>
#include <cmath>
#include <vector>
#include <functional>
#include <cstdlib>

//Platform-specific stuff - abstract this away
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "net.pb.h"
#include "weights.pb.h"
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

using namespace std;

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
        //bias to out
        float b_0, b_1;
        //error for output units
        float d_o_0, d_o_1;
        //error for hidden units
        float d_h_0, d_h_1;
        //float rate = 0.1;
        //derivative of cost funtion w.r.t. outputs
        float d_c_o_0, d_c_o_1;
        //derivative of outputs w.r.t. net
        float d_ao_0, d_ao_1;
        //derivate of hidden activations
        float d_ah_0, d_ah_1; 

        //derivative of cost w.r.t biases
        float d_b_0, d_b_1;

        //Weight gradients
        float cost_grad_o_00;
        float cost_grad_o_01;
        float cost_grad_o_10;
        float cost_grad_o_11;

        float cost_grad_b_0;
        float cost_grad_b_1;

        float cost_grad_h_00;
        float cost_grad_h_01;
        float cost_grad_h_10;
        float cost_grad_h_11;

        yantk::TrainingWeights trainweight_msg;

        std::function<float(float)> act_fn;
        std::function<float(float)> act_deriv_fn;
        std::function<float(float, float)> cost_fn;
        std::function<float(float, float)> cost_deriv_fn;

        void AddWeightsProto(int iteration)
        {
            yantk::IterationWeights* pCurrIterWeights = trainweight_msg.add_iteration_weights();
            pCurrIterWeights->set_iteration(iteration);
            yantk::Weights* pw = new yantk::Weights();
            pw->set_w_h_00(w_h_00);
            pw->set_w_h_01(w_h_01);
            pw->set_w_h_10(w_h_10);
            pw->set_w_h_11(w_h_11);
            pw->set_b_0(b_0);
            pw->set_b_1(b_1);
            pw->set_w_o_00(w_o_00);
            pw->set_w_o_01(w_o_01);
            pw->set_w_o_10(w_o_10);
            pw->set_w_o_11(w_o_11);

            
            pCurrIterWeights->set_allocated_weights(pw);
        }

        void DumpWeights(const string& path)
        {
            int weight_proto_fd = open(path.c_str(), O_WRONLY | O_CREAT);
            google::protobuf::io::FileOutputStream weights_stream(weight_proto_fd);
            google::protobuf::TextFormat::Print(trainweight_msg, &weights_stream);
            weights_stream.Close();
        }

        static float sigmoid(float f)
        {
            return 1.0f / (1.0f + exp(-f));
        }

        static float sigderiv(float f)
        {
            return f * (1.0 - f);
        }

        static float my_tanh(float f)
        {
            return tanh(f);
        }

        static float tanh_deriv(float f)
        {
            return 1 - (f * f);
        }

        static float relu(float f)
        {
            return f > 0.0 ? f : 0.0;
        }

        static float relu_deriv(float f)
        {
            return f > 0.0 ? 1.0 : 0.0;
        }

        void set_act(const std::string& act)
        {
            if(act == "sigmoid")
            {
                act_fn = TestNet::sigmoid;
                act_deriv_fn = TestNet::sigderiv;
            }
            else if(act == "tanh")
            {
                act_fn = TestNet::my_tanh;
                act_deriv_fn = TestNet::tanh_deriv;
            }
            else if(act == "relu")
            {
                act_fn = TestNet::relu;
                act_deriv_fn = TestNet::relu_deriv;
            }

            else
            {
                cout << "Bad: " << act << endl;
                throw std::runtime_error("bad activation fn");
            }

        }

        float errderiv(float t, float o)
        {
            return -(t-o);
        }

        float squared_error(float t, float o)
        {
            return ((t - o) * (t - o) / 2.0);
        }

        void InitializeWeights()
        {
            w_h_00 = 2.4096e-01;
            w_h_01 = -2.2595e-01;
            w_h_10 = -1.8613e-01;
            w_h_11 = 9.4031e-01;
            w_o_00 = -7.0273e-01;
            w_o_01 = -1.7906e-01;
            w_o_10 = -1.4896e-01;
            w_o_11 = 5.9187e-01;
            b_0 = 0.0;
            b_1 = 0.0;
        }

        void dump_weights()
        {
            cout << "W1: " << endl;
            cout << w_h_00 << " " << w_h_01 << endl;
            cout << w_h_10 << " " << w_h_11 << endl;
            cout << "W2: " << endl;
            cout << w_o_00 << " " << w_o_01 << endl;
            cout << w_o_10 << " " << w_o_11 << endl;
            cout << "Biases: " << endl; 
            cout << b_0 << " " << b_1 << endl; 
        }

        void dump_all()
        {

            dump_weights(); 

            cout << "Forward stuff: " << endl;
            cout << "net_h_0: " << net_h_0 << endl;
            cout << "net_h_1: " << net_h_1 << endl;
            cout << "act_h_0: " << h_0 << endl;
            cout << "act_h_1: " << h_1 << endl;
            cout << "net_o_0: " << net_o_0 << endl;
            cout << "net_o_1: " << net_o_1 << endl;
            cout << "act_o_0: " << o_0 << endl;
            cout << "act_o_1: " << o_1 << endl;

            cout << endl << "Error stuff: " << endl;

            cout << "truth y0: " << y0 << ", pred y0 " << o_0 << ", error = " << y0 - o_0 << ", sq_err=" << squared_error(y0, o_0) << endl;
            cout << "truth y1: " << y1 << ", pred y1 " << o_1 << ", error = " << y1 - o_1 << ", sq_err=" << squared_error(y1, o_1) << endl;


            cout << endl << "Backward stuff: " << endl;
            cout << "d_c/o_0: " << d_c_o_0 << endl;
            cout << "d_c/o_1: " << d_c_o_1 << endl;
            cout << "d_ao_0: " << d_ao_0 << endl;
            cout << "d_ao_1: " << d_ao_1 << endl;

            cout << "d_o_0: " << d_o_0 << endl;
            cout << "d_o_1: " << d_o_1 << endl;

            cout << "d_b_0: " << d_b_0 << endl;
            cout << "d_b_1: " << d_b_1 << endl;

            cout << "cost_grad_o_00: " << cost_grad_o_00 << endl;
            cout << "cost_grad_o_01: " << cost_grad_o_01 << endl;
            cout << "cost_grad_o_10: " << cost_grad_o_10 << endl;
            cout << "cost_grad_o_11: " << cost_grad_o_11 << endl;

            cout << "d_ah_0: " << d_ah_0 << endl;
            cout << "d_ah_1: " << d_ah_1 << endl;

            cout << "d_h_0: " << d_h_0 << endl;
            cout << "d_h_1: " << d_h_1 << endl;

            cout << "cost_grad_h_00: " << cost_grad_h_00 << endl;
            cout << "cost_grad_h_01: " << cost_grad_h_01 << endl;
            cout << "cost_grad_h_10: " << cost_grad_h_10 << endl;
            cout << "cost_grad_h_11: " << cost_grad_h_11 << endl;

        }

        void DoUpdates(float rate)
        {
            //Now do the updates.
            w_o_00 -= rate * cost_grad_o_00;
            w_o_01 -= rate * cost_grad_o_01;
            w_o_10 -= rate * cost_grad_o_10;
            w_o_11 -= rate * cost_grad_o_11;

            b_0 -= rate * d_b_0;
            b_1 -= rate * d_b_1;

            w_h_00 -= rate * cost_grad_h_00;
            w_h_01 -= rate * cost_grad_h_01;
            w_h_10 -= rate * cost_grad_h_10;
            w_h_11 -= rate * cost_grad_h_11;
        }

        //Returns error for an example
        float iterate(float set_x0, float set_x1, float set_y0, float set_y1, bool dump, float rate, bool update)
        {
            x0 = set_x0;
            x1 = set_x1;
            y0 = set_y0;
            y1 = set_y1;
            //forward
            net_h_0 = x0 * w_h_00;
            net_h_0 += x1 * w_h_01; 
            net_h_1 = x0 * w_h_10;
            net_h_1 += x1 * w_h_11; 
            h_0 = act_fn(net_h_0);
            h_1 = act_fn(net_h_1);
            net_o_0 = h_0 * w_o_00;
            net_o_0 += h_1 * w_o_01;
            net_o_0 += b_0;
            net_o_1 = h_0 * w_o_10;
            net_o_1 += h_1 * w_o_11;
            net_o_1 += b_1;
            o_0 = act_fn(net_o_0);
            o_1 = act_fn(net_o_1);

            //backward
            d_c_o_0 = errderiv(y0, o_0);
            d_ao_0 = act_deriv_fn(o_0);
            d_o_0 = d_c_o_0  * d_ao_0;
            d_b_0 = d_o_0;
            d_c_o_1 = errderiv(y1, o_1);
            d_ao_1 = act_deriv_fn(o_1);
            d_o_1 = d_c_o_1 * d_ao_1;
            d_b_1 = d_o_1;

            d_ah_0 = act_deriv_fn(h_0);
            float downstream_error_h0 = w_o_00 * d_o_0;
            downstream_error_h0 += w_o_10 * d_o_1;
            d_h_0 = d_ah_0 * downstream_error_h0;
            d_ah_1 = act_deriv_fn(h_1);
            float downstream_error_h1 = w_o_01 * d_o_0;
            downstream_error_h1 += w_o_11 * d_o_1;
            d_h_1 = d_ah_1 * downstream_error_h1;

            cost_grad_o_00 = d_o_0 * h_0;
            cost_grad_o_01 = d_o_0 * h_1;
            cost_grad_o_10 = d_o_1 * h_0;
            cost_grad_o_11 = d_o_1 * h_1;



            cost_grad_h_00 = d_h_0 * x0;
            cost_grad_h_01 = d_h_0 * x1;
            cost_grad_h_10 = d_h_1 * x0;
            cost_grad_h_11 = d_h_1 * x1;

            if(dump)
            {
                cout << "Pre-update weights: " << endl;
                dump_weights();
            }

            if(update)
            {
                DoUpdates(rate);
            }


            if(dump)
            {
                dump_all();
            }

            //error
            float err = squared_error(y0, o_0) + squared_error(y1, o_1);
            return err;
        }

        //Does a forward + backward iteration for all the training examples,
        //returns total error.
        float run_all_examples(bool dump, float rate, bool batch)
        {

            vector<vector<float>> all_xs = {
                {0.0, 0.0}, 
                {0.0, 1.0},
                {1.0, 0.0}, 
                {1.0, 1.0} 
            };
            vector<vector<float>> all_ys = {
                {0.0, 1.0},
                {1.0, 0.0},
                {1.0, 0.0},
                {0.0, 1.0}
            };

            float err = 0.0f;
            if(batch)
            {
                float cost_grad_o_00_sum = 0.0;
                float cost_grad_o_01_sum = 0.0;
                float cost_grad_o_10_sum = 0.0;
                float cost_grad_o_11_sum = 0.0;
                float cost_grad_b_0_sum = 0.0;
                float cost_grad_b_1_sum = 0.0;
                float cost_grad_h_00_sum = 0.0;
                float cost_grad_h_01_sum = 0.0;
                float cost_grad_h_10_sum = 0.0;
                float cost_grad_h_11_sum = 0.0;
                //Accumulate cost_gradients, average for update
                for(int i=0; i < all_xs.size(); ++i)
                {
                    err += iterate(all_xs[i][0], all_xs[i][1], all_ys[i][0], all_ys[i][1], dump, rate, false);
                    cost_grad_o_00_sum += cost_grad_o_00;
                    cost_grad_o_01_sum += cost_grad_o_01;
                    cost_grad_o_10_sum += cost_grad_o_10;
                    cost_grad_o_11_sum += cost_grad_o_11;
                    cost_grad_b_0_sum += cost_grad_b_0;
                    cost_grad_b_1_sum += cost_grad_b_1;
                    cost_grad_h_00_sum += cost_grad_h_00;
                    cost_grad_h_01_sum += cost_grad_h_01;
                    cost_grad_h_10_sum += cost_grad_h_10;
                    cost_grad_h_11_sum += cost_grad_h_11;
                }

                cost_grad_o_00 = cost_grad_o_00_sum / (float)all_xs.size();
                cost_grad_o_01 = cost_grad_o_01_sum / (float)all_xs.size();
                cost_grad_o_10 = cost_grad_o_10_sum / (float)all_xs.size();
                cost_grad_o_11 = cost_grad_o_11_sum / (float)all_xs.size();
                cost_grad_b_0 = cost_grad_b_0_sum / (float)all_xs.size();
                cost_grad_b_1 = cost_grad_b_1_sum / (float)all_xs.size();
                cost_grad_h_00 = cost_grad_o_00_sum / (float)all_xs.size();
                cost_grad_h_01 = cost_grad_o_01_sum / (float)all_xs.size();
                cost_grad_h_10 = cost_grad_o_10_sum / (float)all_xs.size();
                cost_grad_h_11 = cost_grad_o_11_sum / (float)all_xs.size();
                /*
                   cost_grad_o_00 = cost_grad_o_00_sum;
                   cost_grad_o_01 = cost_grad_o_01_sum; 
                   cost_grad_o_10 = cost_grad_o_10_sum;
                   cost_grad_o_11 = cost_grad_o_11_sum;
                   cost_grad_b_0 = cost_grad_b_0_sum;
                   cost_grad_b_1 = cost_grad_b_1_sum;
                   cost_grad_h_00 = cost_grad_o_00_sum;
                   cost_grad_h_01 = cost_grad_o_01_sum;
                   cost_grad_h_10 = cost_grad_o_10_sum;
                   cost_grad_h_11 = cost_grad_o_11_sum;
                   */

                DoUpdates(rate);
                cout << endl << endl << endl << endl << "Post-update dump: " << endl;
                cout << endl << endl << endl <<endl;
                dump_all();

            }
            else
            {
                for(int i=0; i < all_xs.size(); ++i)
                {
                    err += iterate(all_xs[i][0], all_xs[i][1], all_ys[i][0], all_ys[i][1], dump, rate, true);
                }            
            }
            return err;
        }

};

yantk::NetDesc::NonlinearityType StringToDescNonlinearity(const std::string& convertMe)
{


    if(convertMe == "sigmoid") return yantk::NetDesc::SIGMOID;
    if(convertMe == "tanh") return yantk::NetDesc::TANH;
    if(convertMe == "relu") return yantk::NetDesc::RELU;
    throw std::runtime_error("Invalid nonlinearity type"); 
}


string DescNonlinearityToString(yantk::NetDesc::NonlinearityType convertMe)
{

    switch(convertMe)
    {
        case yantk::NetDesc::SIGMOID:
            return "sigmoid";
        case yantk::NetDesc::TANH:
            return "tanh";
        case yantk::NetDesc::RELU:
            return "relu";
        default:
            return "yourmom";
    }
}

int main(int argc, char** argv)
{
    int iters = 0;
    float rate = 0.1;
    string act = "sigmoid";
    bool do_batch = false;
    string output_weights_file;

    if(argc == 2)
    {
        cout << "Single arg, assuming config file" << endl;
        GOOGLE_PROTOBUF_VERIFY_VERSION;
        yantk::NetDesc desc;
        desc.set_num_iterations(10000);
        desc.set_learning_rate(.1);
        desc.set_nonlinearity(yantk::NetDesc::SIGMOID);
        string ser;
        google::protobuf::TextFormat::PrintToString(desc, &ser);
        cout << "set from mem:" << endl;
        cout << ser << endl;

        int fd = open(argv[1], O_RDONLY);
        google::protobuf::io::FileInputStream net_desc_stream(fd);
        google::protobuf::TextFormat::Parse(&net_desc_stream, &desc);
        //close(fd);

        cout << "set from file: " << endl;
        google::protobuf::TextFormat::PrintToString(desc, &ser);
        iters = desc.num_iterations();

        cout << ser << endl;
        act = DescNonlinearityToString(desc.nonlinearity());
            
        rate = desc.learning_rate();
        output_weights_file = desc.output_weights_file();

    }
    else
    {
        cout << "Multiple args, assuming command line args and no config file" << endl;
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
        if(argc > 4)
        {
            if(string(argv[4]) == "batch")
            {
                cout << "Running batch mode" << endl;
                do_batch = true;
            }
        }
    }


    yantk::NetDesc dump_desc;
    dump_desc.set_num_iterations(iters);
    dump_desc.set_learning_rate(rate);
    dump_desc.set_batch(do_batch);
    dump_desc.set_nonlinearity(StringToDescNonlinearity(act));
    int desc_proto_fd = open("thisrun.netproto", O_RDWR | O_CREAT);
    google::protobuf::io::FileOutputStream desc_stream(desc_proto_fd);
    google::protobuf::TextFormat::Print(dump_desc, &desc_stream);
    desc_stream.Close();

    TestNet tn;
    tn.InitializeWeights();

    tn.set_act(act);
    int report_freq = 100000;
    for(int i=0; i < iters; ++i)
    {

        tn.AddWeightsProto(i); 
        /*
        if(i % report_freq == 0)
        {
            tn.AddWeightsProto(i); 
        }
        */
        float err = tn.run_all_examples((i % report_freq) == 0 || i == iters - 1, rate, do_batch);
        //float err = tn.run_all_examples(false, rate);
        if((i % report_freq) == 0 || i == iters - 1)
        {
            cout << "Err for iter " << i << ": " << err << endl;
        }
    }

    tn.DumpWeights(output_weights_file.c_str());
}
