//First cut: just a straightforward feedforward net for mnist.

//Best backprop explanation I've found: http://neuralnetworksanddeeplearning.com/chap2.html

#include <functional>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include <cmath>
#include <cassert>
#include <cblas.h>


template<typename T> void MatMul(const T* A, const T* B, T* C, int m, int n, int k)
{
    throw "NYI";
}

template<> void MatMul<float>(const float* A, const float* B, float* C, int m, int n, int k)
{
    cblas_sgemm(CBLAS_ORDER::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasNoTrans,
        m, n, k, 1.0f, A, k, B, n, 1.0, C, n);
}

template<typename T> class Matrix
{
public:
    int rows;
    int cols;
    std::vector<T> data;
    bool ownMemory;
    std::string name;

    void ReadAscii(const std::string& fileName)
    {
        std::ifstream ifs(fileName);
        if (!ifstream)
        {
            std::ostringstream ostr;
            ostr << "Could not open file " << fileName << " for read";
            throw std::runtime_error(ostr.str().c_str());
        }
        int lastCnt = -1;
        std::string currLine;
        int lineCnt = 0;
        while (std::getline(ifs, currLine))
        {
            ++lineCnt;
            if (curLine[0] == '#') continue;
            std::istringstream istr(currLine);
            int colCnt = 0;
            T curr;
            while (istr)
            {
                istr >> curr;
                data.push_back(curr);
                ++colCnt;
            }
            if (lastCnt == -1)
            {
                lastCnt = colCnt;
            }
            if (colCnt != lastCnt)
            {
                ostringstream err;
                err << "Count of elements in line " << lineCnt << "(" << colCnt <<
                    ") !=  count in prev line(s) (" << lastCnt << ")";
                throw (std::runtime_error(err.str().c_str()));
            }
        }
        cols = lastCnt;
        rows = data.size() / cols;
        std::assert(data.size() % rows == 0);
    }

    Matrix() { rows = 0; cols = 0; ownMemory = false; name = "anonymous"; }

    Matrix(int r, int c, T* d)
    {
        assert(r > 0 && c > 0);
        rows = r;
        cols = c;
        data.insert(data.begin(), d, d + (r*c));
        name = "anonymous";
    }

    Matrix(int r, int c, std::vector<T> d)
    {
        assert(r > 0 && c > 0);
        assert(r * c == d.size());
        rows = r;
        cols = c;
        std::copy(d.begin(), d.end(), std::back_inserter(data));
        name = "anonymous";
    }

    void Multiply(const Matrix<T>& B, Matrix<T>& C) const
    {
        std::cout << "Multiplying A(" << name << ")[" << rows << ":" << cols << "] * B (" << B.name << ")[" << B.rows << ":" << B.cols << "]";
        assert(rows > 0 && B.rows > 0 && cols > 0 && B.cols > 0);
        assert(cols == B.rows);
        if (C.rows != cols || C.cols != B.cols)
        {
            C.rows = rows;
            C.cols = B.cols;
            C.data.resize(C.rows * C.cols);
        }
        MatMul<T>(data.data(), B.data.data(), C.data.data(), rows, B.cols, cols);
    }

    void ApplyEltwiseFn(std::function<T(T)> fn)
    {
        for (int i = 0; i < data.size(); ++i)
        {
            data[i] = fn(data[i]);
        }
    }

    void EltwiseSigmoid()
    {
        ApplyEltwiseFn([](T t) {return (T)1 / ((T)1 + exp(-t)); });
    }

    void EltwiseSigmoidDerivative()
    {
        ApplyEltwiseFn([](T t) {return t * ((T)1 - t);  });
    }

    static Matrix<T> ApplyMemberwiseFn(const Matrix<T>& lhs, const Matrix<T>& rhs, std::function<T(const T lhs, const T rhs)> fn)
    {
        assert(lhs.rows == rhs.rows && lhs.cols == rhs.cols);
        Matrix<T> ret(lhs.rows, rhs.cols, std::vector<T>(lhs.rows * lhs.cols));
        for (int i = 0; i < lhs.data.size(); ++i)
        {
            ret.data[i] = fn(lhs.data[i], rhs.data[i]);
        }
        return ret;
    }

    static Matrix<T> MemberwiseSubtract(const Matrix<T>& lhs, const Matrix<T>& rhs)
    {
        assert(lhs.data.size() == rhs.data.size());
        return ApplyMemberwiseFn(lhs, rhs, [](T l, T r) { return l - r; }); 
    }

    static Matrix<T> MemberwiseMultiply(const Matrix<T>& lhs, const Matrix<T>& rhs)
    {
        assert(lhs.data.size() == rhs.data.size());
        return ApplyMemberwiseFn(lhs, rhs, [](T l, T r) { return l * r; }); 
    }

    static Matrix<T> MemberwiseAdd(const Matrix<T>& lhs, const Matrix<T>& rhs)
    {
        assert(lhs.data.size() == rhs.data.size());
        return ApplyMemberwiseFn(lhs, rhs, [](T l, T r) { return l + r; }); 
    }

    T get(int r, int c) const
    {
        return data[rc2offset(r, c)];
    }
    void set(int r, int c, T t)
    {
        data[rc2offset(r, c)] = t;
    }
    int rc2offset(int r, int c) const
    {
        return (r * cols) + c;
    }

    void Transpose(Matrix<T>& outMatrix) const
    {
        outMatrix.data.resize(data.size());
        outMatrix.rows = cols;
        outMatrix.cols = rows;
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                outMatrix.set(j, i, get(i, j));
            }
        }
    }

    void head(std::ostream& ostr, int numRows=10, int numCols=10) const
    {
        for (int i = 0; i < numRows && i < rows; ++i)
        {
            for (int j = 0; j < numCols && j < cols; ++j)
            {
                ostr << get(i, j) << ' ';
            }
            if (numCols < cols)
            {
                ostr << "...";
            }
            ostr << std::endl;
        }
        if (numRows < rows)
        {
            ostr << "...";
        }
        ostr << std::endl;
    }
};

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


//Not efficient.
template<typename T>void ReadMatrix(std::string fn, Matrix<T>& mat)
{
    std::ifstream ifs(fn);
    std::string currLine;
    T fCurr;
    int numRows = 0;
    int numCols = -1;
    while (std::getline(ifs, currLine))
    {
        int currCols = 0;
        std::istringstream istr(currLine, std::ios_base::in);
        while (istr >> fCurr)
        {
            mat.data.push_back(fCurr);
            ++currCols;
        }
        if (numCols != -1 && currCols != numCols)
        {
            throw std::runtime_error("row count mismatch");
        }
        numCols = currCols;
        ++numRows;
        mat.cols = numCols;
        mat.rows = numRows;
    }
}
/*
def sigmoid(x) :
    return 1 / (1 + np.exp(-x))
    */

template<typename T> T scalarsigmoid(T t)
{
    return (T)1 / ((T)1 + exp(-t));
}

/*
    def forward(x, WEIGHTS, nl) :
    #note that we transpose x to put the training examples in columns here
    inAct = np.transpose(x);
activations = []
for i in range(0, len(WEIGHTS)) :
    W = WEIGHTS[i]
    NET = np.dot(W, inAct)
    if nl == 'sigmoid' :
        act = sigmoid(NET);
elif nl == 'tanh':
act = np.tanh(NET);
elif nl == 'relu':
act = rectifier(NET);
    else:
    raise Exception('unsupported nonlinearity')
        if i != len(WEIGHTS) - 1 :
            act[0, :] = 1
            activations = activations + [act]
            inAct = act
            return np.array(activations)

            def compute_error(A, Y) :
            last = len(A) - 1
            err = Y - A[last]
            sqerr = err * err;
    return np.sum(sqerr)
*/

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


    return 0;
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
    if (argc != 3)
    {
        std::cerr << "Usage: yantk TRAIN TEST" << std::endl;
    }
    //Straightforward port of trainff_gpu.m
    //For now we will continue with the somewhat weird convention there of transposing the inputs so that the examples
    //are in columns instead of in rows, but I don't really see how that's advantageous so will probably switch
    //to the more intuitive setup once I've verified that it works.
    Matrix<float> TRAIN;
    TRAIN.ReadAscii(argv[1]);
    Matrix<float> TEST;
    TEST.ReadAscii(argv[2]);

    //TODO: Read in from file(s)
    float learn_rate = 0.1;
    float momentum = 0.0;
    int num_layers = 3;
    int minibatch_size = 1;
    bool scaleFeatures = false;
    std::vector<Matrix<float>> WEIGHTS;
    bool init_weights = true;
    float learn_rate = 0.1;

    int trainRowCnt = TRAIN.rows;
    int testRowCnt = TEST.rows;
    int colCnt = TRAIN.cols;
    int featCnt = colCnt;

    float SCALERATIO = -100;
    float MINUNSCALED = -100;

    TR_X = GetFeatures(TRAIN);
    TE_X = GetFeatures(TEST);




}

