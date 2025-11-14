#ifndef RICCATI_KERNELS_CUH
#define RICCATI_KERNELS_CUH

#include <cuda_runtime.h>

/**
 * Riccati recursion kernels for LQR
 * Used for validation and comparison with BCR
 */

// Backward pass: Compute P[k], p[k], K[k], k[k]
// P[k] = Q[k] + A[k]^T * P[k+1] * A[k] - K[k]^T * H[k] * K[k]
// where K[k] = -H[k]^{-1} * G[k], H[k] = R + B^T*P[k+1]*B, G[k] = B^T*P[k+1]*A
__global__ void riccatiBackwardKernel(
    const double* __restrict__ Q,    // [T+1, n, n]
    const double* __restrict__ R,    // [T, m, m]
    const double* __restrict__ A,    // [T, n, n]
    const double* __restrict__ B,    // [T, n, m]
    const double* __restrict__ q,    // [T+1, n]
    const double* __restrict__ r,    // [T, m]
    double* __restrict__ P,          // [T+1, n, n] (output)
    double* __restrict__ p,          // [T+1, n] (output)
    double* __restrict__ K,          // [T, m, n] (output)
    double* __restrict__ k_ff,       // [T, m] (output)
    int nx,
    int nu,
    int time_step
);

// Forward pass: Rollout using feedback gains
// x[k+1] = A[k]*x[k] + B[k]*u[k]
// u[k] = K[k]*x[k] + k[k]
__global__ void riccatiForwardKernel(
    const double* __restrict__ A,
    const double* __restrict__ B,
    const double* __restrict__ K,
    const double* __restrict__ k_ff,
    const double* __restrict__ x0,
    double* __restrict__ x,          // [T+1, n] (output)
    double* __restrict__ u,          // [T, m] (output)
    int nx,
    int nu,
    int horizon
);

// Compute Riccati gains at a single time step
__device__ void computeRiccatiGains(
    const double* Q, const double* R,
    const double* A, const double* B,
    const double* P_next, const double* p_next,
    double* P, double* p,
    double* K, double* k_ff,
    int nx, int nu
);

#endif // RICCATI_KERNELS_CUH