//First cut: just a straightforward feedforward net for mnist.

//Best backprop explanation I've found: http://neuralnetworksanddeeplearning.com/chap2.html

#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include <cmath>
#include <exception>
#include "matrix.h"

using namespace std;

template<typename T> void RandInitMatrix(Matrix<T>& randomizeMe, T min, T max)
{
    for (int i = 0; i < randomizeMe.data.size(); ++i)
    {
        T r = (T)rand() / (T)RAND_MAX;
        T diff = max - min;
        r *= diff;
        randomizeMe.data[i] = min + r;
    }
}

template<typename T> void forward(Matrix<T>& input, const std::vector<Matrix<T>>& weights, /*out*/ 
        std::vector<Matrix<T>>& weightedSums, std::vector<Matrix<T>>& activations)
{
    weightedSums.clear();
    activations.clear();
    //activations.push_back(input);
    //TODO: pass in transposed matrix
    //Matrix<T> transIn;
    //input.Transpose(transIn);
    //assert(transIn.cols == weights[0].rows);
    //Column 1 of the original matrix (row 1 in transposed mat) will have the biases, which should be set to 1.
    Matrix<T>& act = input;
    for (int i = 0; i < weights.size(); ++i)
    {
        //set biases
        //for (int i = 0; i < act.rows; ++i) { act.data[i] = (T)1; }
        Matrix<T> newAct;
        act.Multiply(weights[i], newAct);
        //std::cout << "Fwd vals: " << std::endl;
        //newAct.head(std::cout);
        weightedSums.push_back(newAct);
        newAct.EltwiseSigmoid();
        //std::cout << "Fwd acts: " << std::endl;
        //newAct.head(std::cout);
        activations.push_back(newAct);
        act = newAct;
    }
}


void ff_scratch_network()
{
    //XOR problem without bias from https://stevenmiller888.github.io/mind-how-to-build-a-neural-network/
    Matrix<float> X (4, 2, std::vector<float>{ 0, 0, 0, 1.0f, 1.0f, 0, 1.0f, 1.0f });
    X.name = "X";
    Matrix<float> Y (4, 1, std::vector<float>{ 0, 1, 1, 0 });
    Y.name = "Y";
    Matrix<float> WHI(2, 3, std::vector<float>{ 0.8f, 0.4f, 0.3f, 0.2f, 0.9f, 0.5f });
    WHI.name = "WHI";
    Matrix<float> WOH(3, 1, std:: vector<float>{0.3f, 0.5f, 0.9f});
    WHI.name = "WOH";

    std::vector<Matrix<float>> allWeights = { WHI, WOH };
    std::vector<Matrix<float>> weightedSums, activations;
    forward(X, allWeights, weightedSums, activations);

    for (int i = 0; i < (int)weightedSums.size(); ++i)
    {
        std::cout << "Layer " << i << std::endl;
        std::cout << "Weighted sums: " << std::endl;
        weightedSums[i].head(std::cout);
        activations[i].head(std::cout);
    }

    Matrix<float> err = Matrix<float>::MemberwiseSubtract(Y, activations[activations.size() - 1]);
    err.name = "err";

    std::cout << "Err: " << std::endl;
    err.head(std::cout);


    //Now do backprop
    //First we compute d_k, which for the last layer is the derivative of the cost function with respect to the
    //activations of the layer times the derivative of the nonlinearity function.
    //For quadratic error the first term is just (a_k - y), and the second term is 
    //the derivative of the sigmoid function on z_k (nlderiv(z_k)
    std::vector<Matrix<float>> ds(allWeights.size());

    int numLayers = weightedSums.size();
    Matrix<float> dcda = Matrix<float>::MemberwiseSubtract(activations[numLayers - 1], Y);
    dcda.name = "dcda";
    Matrix<float> nlDeriv = weightedSums[numLayers - 1];
    nlDeriv.name = "nlDeriv";
    nlDeriv.EltwiseSigmoidDerivative();
    Matrix<float> dLast = Matrix<float>::MemberwiseMultiply(dcda, nlDeriv);
    dLast.name = "dLast";
    std::cout << "DLast: " << std::endl;
    dLast.head(std::cout);
    ds[numLayers - 1] = dLast;

    //For all of the subsequent layers we compute d^l, as:
    //d_l = (w(l+1)^T * d^l+1) memberwise_times nlderiv(z_l)
    int currIdx = numLayers - 2;
    while (currIdx >= 0)
    {
        Matrix<float> transWeights;
        transWeights.name = "transWeights";
        allWeights[currIdx + 1].Transpose(transWeights);
        Matrix<float> term1;
        term1.name = "term1";
        ds[currIdx + 1].Multiply(transWeights, term1);
        Matrix<float> term2 = weightedSums[currIdx];
        term2.name = "term2";
        term2.EltwiseSigmoidDerivative();
        Matrix<float> dl = Matrix<float>::MemberwiseMultiply(term1, term2);
        dl.name = "dl";
        ds[currIdx] = dl;
        --currIdx;
    }

    /*
       Matrix<float> images;
       Matrix<float> labels;
       ReadMatrix("c:\\users\\anthaue\\onedrive\\documents\\school\\mnist\\images.10.mat",
       images);

       ReadMatrix("c:\\users\\anthaue\\onedrive\\documents\\school\\mnist\\labels.10.mat",
       labels);

       std::cout << "Images is " << images.rows << " x " << images.cols << std::endl;
       images.head(std::cout);
       std::cout << "Lablels is " << labels.rows << " x " << labels.cols << std::endl;
       labels.head(std::cout);
       */


}


class Layer
{
    public:
        enum {INPUT, FC, SOFTMAX} LayerType;
        void Forward();
        void ComputeGradient(float* prevLayerOut);

    private:
        Layer* m_pInput;
        int m_outputSize;
        float* m_pOutput;
        float* m_pActivations;

};

class Optimizer //e.g. SGD, AdaGrad, ADAM, etc.
{
    void iterate();
};


int main(int argc, char** argv)
{
    std::cout << "Starting up" << std::endl;
    srand(42);
    if (argc != 3)
    {
        std::cerr << "Usage: yantk TRAIN TEST" << std::endl;
        return 1;
    }
    //Straightforward port of trainff_gpu.m
    //For now we will continue with the somewhat weird convention there of transposing the inputs so that the examples
    //are in columns instead of in rows, but I don't really see how that's advantageous so will probably switch
    //to the more intuitive setup once I've verified that it works.
    Matrix<float> train;
    std::ifstream ifsTrain(argv[1]);
    train.ReadAscii(ifsTrain);
    Matrix<float> test;
    std::ifstream ifsTest(argv[2]);
    test.ReadAscii(ifsTest);

    std::cout << "Read train and test." << std::endl;

    std::cout << "train" << std::endl;
    train.head(cout);
    cout << "test" << endl;
    test.head(cout);

    //TODO: Read in from file(s)
    float learn_rate = 0.1;
    float momentum = 0.0;
    int num_layers = 3;
    int minibatch_size = 1;
    bool scaleFeatures = false;
    std::vector<Matrix<float>> WEIGHTS;
    bool init_weights = true;

    int trainRowCnt = train.rows;
    int testRowCnt = test.rows;
    int colCnt = train.cols;
    int featCnt = colCnt;

    float SCALERATIO = -100;
    float MINUNSCALED = -100;

    //TR_X = GetFeatures(TRAIN);
    //TE_X = GetFeatures(TEST);
    cout << "Slicing into x and y..." << std::endl;
    Matrix<float> tr_x_pre;
    train.GetSubmatrix(0, train.rows - 1, 0, train.cols - 1, tr_x_pre);
    Matrix<float> te_x_pre;
    train.GetSubmatrix(0, test.rows - 1, 0, test.cols - 1, te_x_pre);

    //Add bias column

    Matrix<float> trainOnes;
    Matrix<float>::GetSingleValMatrix(tr_x_pre.rows, 1, 1, trainOnes);
    Matrix<float> testOnes;
    Matrix<float>::GetSingleValMatrix(te_x_pre.rows, 1, 1, testOnes);

    Matrix<float> tr_x;
    Matrix<float>::ConcatCols(trainOnes, tr_x_pre, tr_x);
    Matrix<float> te_x;
    Matrix<float>::ConcatCols(testOnes, te_x_pre, te_x);

    Matrix<float> tr_y;
    train.GetSubmatrix(train.rows - 1, 1, 0, train.cols, tr_y);
    Matrix<float> te_y;
    train.GetSubmatrix(test.rows - 1, 1, 0, test.cols, te_y);

    //TODO: Most of the rest of this should go in network and/or opt
    //Initialize weights
    enum ActivationType { SIGMOID, TANH, RELU };
    ActivationType at;
    at = SIGMOID;
    float range_min;
    float range_max;
    switch(at)
    {
        case SIGMOID:
        case RELU:
            range_min = -0.05f;
            range_max = 0.05f;
            break;
        case TANH:
            range_min = -0.2f;
            range_max = 0.2f;
            break;
        default:
            throw std::runtime_error("Unknown activation type");
    }

    int numlayers = 3;
    int numHiddenLayers=1;
    int hiddenWidth = 128;
    int numOut = tr_y.Max() + 1;

    std::vector<Matrix<float>> weights;

    cout << "Initializing weight matrices..." << std::endl;

    Matrix<float> in2hid1Weights(hiddenWidth, train.cols);
    Matrix<float>::RandInitMatrix(range_min, range_max, in2hid1Weights); 
    weights.push_back(in2hid1Weights);

    //Hidden layers
    for(int i=1; i < numHiddenLayers; ++i)
    {
        Matrix<float> hid2hid(hiddenWidth, hiddenWidth);
        Matrix<float>::RandInitMatrix(range_min, range_max, hid2hid); 
        weights.push_back(hid2hid);
    }

    //Last hidden -> output
    Matrix<float> hid2out(numOut, hiddenWidth);
    Matrix<float>::RandInitMatrix(range_min, range_max, hid2out); 
    weights.push_back(hid2out);

    std::vector<Matrix<float>> updates_prev;
    std::vector<Matrix<float>> updates_squared_sum;

    cout << "Initializing matrices for statistics" << endl;

    for(int i=0; i < weights.size(); ++i)
    {
        Matrix<float> currPrev;
        Matrix<float>::GetSingleValMatrix(weights[i].rows, weights[i].cols, 0.0f, currPrev);
        updates_prev.push_back(currPrev);
        Matrix<float> currSquaredSum;
        Matrix<float>::GetSingleValMatrix(weights[i].rows, weights[i].cols, 1.0f, currPrev);
        updates_squared_sum.push_back(currSquaredSum);
    }

    cout << "Expanding y matrices..." << endl;

    Matrix<float> target;
    Matrix<float>::GetSingleValMatrix(numOut, trainRowCnt, 0.0f, target);
    for(int i=0; i < trainRowCnt; ++i)
    {
        target.set(tr_y.get(i, 1), i, 1);
    }

    int curr_iter = 0;
    float best_test_acc = -1.0f;

    learn_rate = learn_rate / minibatch_size;

    int iterations = 2;
    while(curr_iter < iterations)
    {
        int right = 0;
        int wrong = 0;
        ++curr_iter;

    }






}

