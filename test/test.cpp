#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "matrix.h"
#include "MSELoss.h"
#include <sstream>

TEST_CASE ("Matrix class", "[Matrix]") 
{

    float testLhs[] = 
    { 
            1.0f, 2.0f,
            3.0f, 4.0f
    };
    float testRhs[] = 
    {
            5.0f, 6.0f,
            7.0f, 8.0f
    };
    Matrix<float> lhsMatrix(2, 2, testLhs);
    Matrix<float> rhsMatrix(2, 2, testRhs);

    SECTION ("Memberwise add")
    {

            Matrix<float> retMatrix = Matrix<float>::MemberwiseAdd(lhsMatrix, rhsMatrix);
            REQUIRE(retMatrix.rows == 2);
            REQUIRE(retMatrix.cols == 2);
            REQUIRE(retMatrix.get(0, 0) == 6.0f);
            REQUIRE(retMatrix.get(0, 1) == 8.0f);
            REQUIRE(retMatrix.get(1, 0) == 10.0f);
            REQUIRE(retMatrix.get(1, 1) == 12.0f);
    }

    SECTION ("Memberwise sub")
    {

            Matrix<float> retMatrix = Matrix<float>::MemberwiseSubtract(lhsMatrix, rhsMatrix);
            REQUIRE(retMatrix.rows == 2);
            REQUIRE(retMatrix.cols == 2);
            REQUIRE(retMatrix.get(0, 0) == -4.0f);
            REQUIRE(retMatrix.get(0, 1) == -4.0f);
            REQUIRE(retMatrix.get(1, 0) == -4.0f);
            REQUIRE(retMatrix.get(1, 1) == -4.0f);
    }

    SECTION ("Memberwise mult")
    {

            Matrix<float> retMatrix = Matrix<float>::MemberwiseMultiply(lhsMatrix, rhsMatrix);
            REQUIRE(retMatrix.rows == 2);
            REQUIRE(retMatrix.cols == 2);
            REQUIRE(retMatrix.get(0, 0) == 5.0f);
            REQUIRE(retMatrix.get(0, 1) == 12.0f);
            REQUIRE(retMatrix.get(1, 0) == 21.0f);
            REQUIRE(retMatrix.get(1, 1) == 32.0f);
    }

    SECTION ("Transpose")
    {
        Matrix<float> trans;
        lhsMatrix.Transpose(trans);
        REQUIRE(trans.rows == 2);
        REQUIRE(trans.cols == 2);
        REQUIRE(trans.get(0, 0) == 1.0f);
        REQUIRE(trans.get(0, 1) == 3.0f);
        REQUIRE(trans.get(1, 0) == 2.0f);
        REQUIRE(trans.get(1, 1) == 4.0f);
    }

    SECTION ("GetSubmatrix")
    {
        Matrix<float> sub;
        lhsMatrix.GetSubmatrix(0, 1, 0, 1, sub);
        REQUIRE(sub.rows == 1);
        REQUIRE(sub.cols == 1);
        REQUIRE(sub.get(0, 0) == 1.0f);

        lhsMatrix.GetSubmatrix(0, 1, 0, 2, sub);
        REQUIRE(sub.rows == 1);
        REQUIRE(sub.cols == 2);
        REQUIRE(sub.get(0, 0) == 1.0f);
        REQUIRE(sub.get(0, 1) == 2.0f);

        lhsMatrix.GetSubmatrix(1, 1, 1, 1, sub);
        REQUIRE(sub.rows == 1);
        REQUIRE(sub.cols == 1);
        REQUIRE(sub.get(0, 0) == 4.0f);

        lhsMatrix.GetSubmatrix(0, 2, 0, 1, sub);
        REQUIRE(sub.rows == 2);
        REQUIRE(sub.cols == 1);
        REQUIRE(sub.get(0, 0) == 1.0f);
        REQUIRE(sub.get(1, 0) == 3.0f);
    }

    SECTION ("ReadAscii")
    {
            const char* rep = 
                    "1.0 2.0\n3.0 4.0";
            std::istringstream s(rep);
            Matrix<float> readToMe;
            readToMe.ReadAscii(s);

            REQUIRE(readToMe.rows == 2);
            REQUIRE(readToMe.cols == 2);
            REQUIRE(readToMe.get(0, 0) == 1.0f);
            REQUIRE(readToMe.get(0, 1) == 2.0f);
            REQUIRE(readToMe.get(1, 0) == 3.0f);
            REQUIRE(readToMe.get(1, 1) == 4.0f);
    }

    SECTION ("ConcatCols")
    {
        Matrix<float> concat;
        Matrix<float>::ConcatCols(lhsMatrix, rhsMatrix, concat);
        REQUIRE(concat.rows == 2);
        REQUIRE(concat.cols == 4);
        REQUIRE(concat.get(0, 0) == 1.0f);
        REQUIRE(concat.get(0, 2) == 5.0f);
        REQUIRE(concat.get(1, 0) == 3.0f);
        REQUIRE(concat.get(1, 3) == 8.0f);
    }

    SECTION ("SingleVal")
    {
        Matrix<float> single;
        Matrix<float>::GetSingleValMatrix(2, 2, 1.0f, single);
        REQUIRE(single.rows == 2);
        REQUIRE(single.cols == 2);
        REQUIRE(single.get(0, 0) == 1.0f);
        REQUIRE(single.get(0, 1) == 1.0f);
        REQUIRE(single.get(1, 0) == 1.0f);
        REQUIRE(single.get(1, 1) == 1.0f);
    }

    SECTION ("Randomize")
    {
        Matrix<float> randomized(2, 2);
        Matrix<float>::RandInitMatrix(-0.5f, 3.0f, randomized);
        for(int i=0; i < randomized.data.size(); ++i)
        {
            REQUIRE(randomized.data[i] >= -0.5f);
            REQUIRE(randomized.data[i] <= 3.0f);
        }
    }

    SECTION ("Min")
    {
        REQUIRE(lhsMatrix.Min() == 1.0f);
        REQUIRE(rhsMatrix.Min() == 5.0f);        
    }

    SECTION ("Max")
    {
        REQUIRE(lhsMatrix.Max() == 4.0f);
        REQUIRE(rhsMatrix.Max() == 8.0f);        
    }

}


TEST_CASE ("Losses", "[LossTest]") 
{

    float testLhs[] = 
    { 
            1.0f, 2.0f,
            3.0f, 4.0f
    };
    float testRhs[] = 
    {
            5.0f, 6.0f,
            7.0f, 8.0f
    };
 
    Matrix<float> lhsMatrix(2, 2, testLhs);
    Matrix<float> rhsMatrix(2, 2, testRhs);

    SECTION ("LossDeriv")
    {
        Matrix<float> retMatrix(2, 2);
        MeanSquaredErrorLoss::LossDerivative(lhsMatrix, rhsMatrix, retMatrix);
        REQUIRE(retMatrix.get(0,0) == 4.0f);
        REQUIRE(retMatrix.get(0,1) == 4.0f);
        REQUIRE(retMatrix.get(1,0) == 4.0f);
        REQUIRE(retMatrix.get(1,1) == 4.0f);
    }




}
