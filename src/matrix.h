#include <fstream>
#include <sstream>
#include <functional>
#include <cblas.h>
#include <cassert>
#include <limits>

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
    std::string name;

    void ReadAscii(std::istream& ifs)
    {
        if (!ifs)
        {
            throw std::runtime_error("bad stream");
        }
        int lastCnt = -1;
        std::string currLine;
        int lineCnt = 0;
        while (std::getline(ifs, currLine))
        {
            ++lineCnt;
            if (currLine[0] == '#') continue;
            std::istringstream istr(currLine);
            int colCnt = 0;
            T curr;
            while (istr && !istr.eof())
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
                std::ostringstream err;
                err << "Count of elements in line " << lineCnt << "(" << colCnt <<
                    ") !=  count in prev line(s) (" << lastCnt << ")";
                throw (std::runtime_error(err.str().c_str()));
            }
        }
        cols = lastCnt;
        rows = data.size() / cols;
        assert(data.size() % rows == 0);
    }

    static void RandInitMatrix(T min, T max, Matrix<T>& randomizeMe)
    {
            for (int i = 0; i < randomizeMe.data.size(); ++i)
            {
                    T r = (T)rand() / (T)RAND_MAX;
                    T diff = max - min;
                    r *= diff;
                    randomizeMe.data[i] = min + r;
            }
    }

    Matrix() { rows = 0; cols = 0; name = "anonymous"; }

    Matrix(int r, int c, T* d)
    {
        assert(r > 0 && c > 0);
        rows = r;
        cols = c;
        data.insert(data.begin(), d, d + (r*c));
        name = "anonymous";
    }

    Matrix(int r, int c)
    {
        rows = r;
        cols = c;
        data.resize(r * c);
        name="anonymous";
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
        std::cout << "Multiplying A(" << name << ")[" << rows << ":" << cols << "] * B (" << B.name << ")[" << B.rows << ":" << B.cols << "]" << std::endl;
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

    void EltwiseTanh()
    {
        ApplyEltwiseFn([](T t) {return tanh(t);});
    }

    void EltwiseSigmoid()
    {
        ApplyEltwiseFn([](T t) {return (T)1 / ((T)1 + exp(-t)); });
    }

    void EltwiseSigmoidDerivative()
    {
        ApplyEltwiseFn([](T t) {return t * ((T)1 - t);  });
    }

    T Min()
    {
        T min = std::numeric_limits<T>::max();
        for(T t : data)
        {
            if(t < min) min = t;
        }
        return min;
    }   

    T Max()
    {
        T max = std::numeric_limits<T>::min();
        for(T t : data)
        {
            if(t > max) max = t;
        }   
        return max;
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

    void GetSubmatrix(int rowBegin, int rowCnt, int colBegin, int colCnt, 
        Matrix<T>& out)
    {
        assert(rowCnt > 0); 
        assert(colCnt > 0); 
        assert(rowBegin >= 0);
        assert(rowBegin >= 0);
        out.rows=rowCnt;
        out.cols=colCnt;
        out.data.resize(rowCnt * colCnt);
        for(int i=rowBegin; i < rowBegin + rowCnt; ++i)
        {
            for(int j=colBegin; j < colBegin + colCnt; ++j)
            {
               out.set(i - rowBegin, j - colBegin, get(i, j)); 
            }
        }
    }

    //Return a new matrix which is the concatenation of the left and right sides
    static void ConcatCols(const Matrix<T>& lhs, const Matrix<T>& rhs, 
                    Matrix<T>& out)
    {
        assert(lhs.rows == rhs.rows);
        out.rows = lhs.rows;
        out.cols = lhs.cols + rhs.cols;
        out.data.resize(out.rows * out.cols);
        //TODO: Make this more efficient using memcpy
        for(int i = 0; i < out.rows; ++i)
        {
                //lhs
                for(int j=0; j < lhs.cols; ++j)
                {
                        out.set(i, j, lhs.get(i, j));
                }
                //rhs
                for(int j=0; j < rhs.cols; ++j)
                {
                        out.set(i, j + lhs.cols, rhs.get(i, j));
                }
        }
    }

    static void GetSingleValMatrix(int rows, int cols, T singleVal, Matrix<T>& out)
    {
        out.rows = rows;
        out.cols = cols;
        out.data.resize(rows * cols);

        for(int i=0; i < out.data.size(); ++i)
        {
                out.data[i] = singleVal;
        }
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

