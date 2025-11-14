#ifndef BLOCK_OPERATIONS_CUH
#define BLOCK_OPERATIONS_CUH

#include <cuda_runtime.h>

/**
 * CUDA kernels for block matrix operations in BCR
 */

// Block Cholesky decomposition (in-place)
__device__ void blockCholesky(double* A, int n);

// Solve L*L^T*x = b using Cholesky factor L
__device__ void blockCholeskySolve(const double* L, const double* b, double* x, int n);

// Invert using Cholesky decomposition
__global__ void blockCholeskyInvert(
    const double* __restrict__ A,
    double* __restrict__ A_inv,
    int block_size,
    int n_blocks
);

// Block matrix multiplication: C = A * B
__global__ void blockMatMul(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    int block_size,
    int n_blocks
);

// Block matrix addition: C = alpha*A + beta*B
__global__ void blockMatAdd(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    double alpha,
    double beta,
    int block_size,
    int n_blocks
);

// BCR forward reduction kernel
// Computes: Q'[i] = Q[i] - B[i-1]^T * Q[i-1]^{-1} * B[i-1] - B[i]^T * Q[i+1]^{-1} * B[i]
__global__ void bcrForwardReduction(
    const double* __restrict__ Q_in,
    const double* __restrict__ B_in,
    const double* __restrict__ q_in,
    double* __restrict__ Q_out,
    double* __restrict__ B_out,
    double* __restrict__ q_out,
    int block_size,
    int stride,
    int n_blocks
);

// BCR backward substitution kernel
// Solves: x[i] = Q[i]^{-1} * (q[i] - B[i-1]^T*x[i-1] - B[i]^T*x[i+1])
__global__ void bcrBackwardSubstitution(
    const double* __restrict__ Q,
    const double* __restrict__ B,
    const double* __restrict__ q,
    double* __restrict__ x,
    int block_size,
    int stride,
    int n_blocks
);

// Helper: Compute Q^{-1} * B efficiently
__global__ void solveQinvB(
    const double* __restrict__ Q,
    const double* __restrict__ B,
    double* __restrict__ result,
    int block_size,
    int n_blocks
);

#endif // BLOCK_OPERATIONS_CUH