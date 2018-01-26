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


enum ActivationType { SIGMOID, TANH, RELU };

void squish(Matrix<float>& squishMe, ActivationType type)
{
    switch(type)
    {
        case SIGMOID:
            squishMe.EltwiseSigmoid();
        case TANH:
            squishMe.EltwiseTanh();
        case RELU:
        default:
            throw std::runtime_error("nyi");
    }
}

vector<Matrix<float>> forward(vector<Matrix<float>>& weights, Matrix<float> x, ActivationType type)
{
    //TODO: we should not be allocating here, but I'll leave that
    //for a refactoring
    vector<Matrix<float>> ret(weights.size());
    Matrix<float> firstResult(x.rows, weights[0].cols);
    weights[0].Multiply(x, firstResult);
    squish(firstResult, type);
    ret.push_back(firstResult);
    for(int i=1; i < weights.size(); ++i)
    {
        Matrix<float> tmp = ret[i-1];
        //Set biases to 1.
        //TODO: Write a matrix method for this, like SetSubmatrix or something
        for(int i=0; i < tmp.rows; ++i)
        {
            tmp.set(i, 0, 1.0f);
        }
        //TODO: implement dropout here
        Matrix<float> nextOut(weights[i].rows, tmp.cols);
        weights[i].Multiply(weights[i-1], tmp); 
        squish(tmp, type);
        ret.push_back(tmp);
    }
}


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
    train.GetSubmatrix(train.rows - 2, 1, 0, train.cols, tr_y);
    Matrix<float> te_y;
    train.GetSubmatrix(test.rows - 2, 1, 0, test.cols, te_y);

    //TODO: Most of the rest of this should go in network and/or opt
    //Initialize weights
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

    //target is transposed, i.e. there is a column per training row.
    Matrix<float> target;
    Matrix<float>::GetSingleValMatrix(numOut, trainRowCnt, 0.0f, target);
    for(int i=0; i < trainRowCnt; ++i)
    {
        target.set(tr_y.get(i, 1), i, 1);
    }

    int curr_iter = 0;
    float best_test_acc = -1.0f;

    learn_rate = learn_rate / minibatch_size;

    int epochs = 2;
    while(curr_iter < epochs)
    {
        int right = 0;
        int wrong = 0;
        ++curr_iter;
        //Read in a minibatch
        int row = 0;
        while(row + minibatch_size < trainRowCnt)
        {
            Matrix<float> xBatch;
            tr_x.GetSubmatrix(row, minibatch_size, 0, tr_x.cols, xBatch);
            Matrix<float> yBatch;
            target.GetSubmatrix(0,tr_y.rows, row, minibatch_size, yBatch);
        }

    }

}

