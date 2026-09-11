#include "mex.h"
#include <cmath>

// Fused vector-query nonstationary squared-exponential kernel and its
// query gradient. Inputs are ordinary real doubles:
//   X[dx,n], ell_train[dx,n], ell_train_sq[dx,n], x[dx,1],
//   ell_query[dx,1], dell_query_dt[dx,1], sigma_f_sq scalar.
// Outputs:
//   Ktx[n,1], and optionally dk_dquery[n,dx].
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 7) {
        mexErrMsgIdAndTxt("GPFM:gpKernelMex:Inputs", "Expected seven inputs.");
    }
    for (int i = 0; i < 7; ++i) {
        if (!mxIsDouble(prhs[i]) || mxIsComplex(prhs[i])) {
            mexErrMsgIdAndTxt("GPFM:gpKernelMex:Type", "All inputs must be real doubles.");
        }
    }
    const mwSize dx = mxGetM(prhs[0]);
    const mwSize n = mxGetN(prhs[0]);
    if (mxGetM(prhs[1]) != dx || mxGetN(prhs[1]) != n ||
        mxGetM(prhs[2]) != dx || mxGetN(prhs[2]) != n ||
        mxGetNumberOfElements(prhs[3]) != dx ||
        mxGetNumberOfElements(prhs[4]) != dx ||
        mxGetNumberOfElements(prhs[5]) != dx ||
        mxGetNumberOfElements(prhs[6]) != 1) {
        mexErrMsgIdAndTxt("GPFM:gpKernelMex:Shape", "Input dimensions do not agree.");
    }

    const double *X = mxGetDoubles(prhs[0]);
    const double *ellTrain = mxGetDoubles(prhs[1]);
    const double *ellTrainSq = mxGetDoubles(prhs[2]);
    const double *x = mxGetDoubles(prhs[3]);
    const double *ellQuery = mxGetDoubles(prhs[4]);
    const double *dellQuery = mxGetDoubles(prhs[5]);
    const double sigmaFSq = mxGetScalar(prhs[6]);

    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
    double *K = mxGetDoubles(plhs[0]);
    double *dK = nullptr;
    if (nlhs > 1) {
        plhs[1] = mxCreateDoubleMatrix(n, dx, mxREAL);
        dK = mxGetDoubles(plhs[1]);
    }

    for (mwSize col = 0; col < n; ++col) {
        double amplitude = 1.0;
        double exponentSum = 0.0;
        double logAmplitudeDtSum = 0.0;
        double exponentLengthDtSum = 0.0;
        for (mwSize dim = 0; dim < dx; ++dim) {
            const mwSize idx = dim + col * dx;
            const double delta = X[idx] - x[dim];
            const double ellQ = ellQuery[dim];
            const double ellSumSq = ellTrainSq[idx] + ellQ * ellQ;
            const double ellSq = 0.5 * ellSumSq;
            amplitude *= std::sqrt((2.0 * ellTrain[idx] * ellQ) / ellSumSq);
            exponentSum += (delta * delta) / ellSq;
            if (dK != nullptr) {
                const double dell = dellQuery[dim];
                logAmplitudeDtSum += (dell / ellQ) -
                    (2.0 * ellQ * dell) / ellSumSq;
                exponentLengthDtSum +=
                    ((delta * delta) * (ellQ * dell)) / (ellSq * ellSq);
            }
        }
        const double kernel = sigmaFSq * amplitude * std::exp(-0.5 * exponentSum);
        K[col] = kernel;
        if (dK != nullptr) {
            for (mwSize dim = 0; dim < dx; ++dim) {
                const mwSize idx = dim + col * dx;
                const double delta = X[idx] - x[dim];
                const double ellQ = ellQuery[dim];
                const double ellSq = 0.5 * (ellTrainSq[idx] + ellQ * ellQ);
                dK[col + dim * n] = (delta / ellSq) * kernel;
            }
            const double deltaT = X[col * dx] - x[0];
            const double ellQT = ellQuery[0];
            const double ellSqT = 0.5 * (ellTrainSq[col * dx] + ellQT * ellQT);
            dK[col] = (deltaT / ellSqT +
                0.5 * exponentLengthDtSum +
                0.5 * logAmplitudeDtSum) * kernel;
        }
    }
}
