#include "matrix.h"

class MeanSquaredErrorLoss
{
    public:
        static void LossDerivative(Matrix<float>& computed, Matrix<float>& gold, Matrix<float>& ret)
        {
            ret = gold.MemberwiseSubtract(gold, computed); 
        }
};
