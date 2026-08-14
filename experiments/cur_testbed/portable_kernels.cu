// Standalone testbed PTX bundle. The generation kernels come directly from
// the challenge source; leverage supplies the Gaussian sketch and norm
// kernels used by the sophisticated selector.
#include "../../tig-challenges/src/cur_decomposition/kernels.cu"
#include "../../tig-algorithms/src/cur_decomposition/leverage/kernels.cu"

// Scale row i of a column-major (rows x cols) matrix by scales[i].
extern "C" __global__ void scale_rows_kernel(
    float *mat, const float *scales, int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int i = idx % rows;
    mat[idx] *= scales[i];
}

// Scale column j of a column-major (rows x cols) matrix by scales[j].
extern "C" __global__ void scale_cols_kernel(
    float *mat, const float *scales, int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int j = idx / rows;
    mat[idx] *= scales[j];
}
