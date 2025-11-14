#include "bcr_solver.cuh"
#include "block_operations.cuh"
#include "riccati_kernels.cuh"
#include <cmath>
#include <cstring>
#include <iostream>

CUDABCRSolver::CUDABCRSolver(const BCRConfig& config) 
    : config_(config), cublas_handle_(nullptr) {
    
    // Initialize cuBLAS
    cublasStatus_t status = cublasCreate(&cublas_handle_);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to create cuBLAS handle");
    }
    
    // Allocate workspace
    BCRStatus bcr_status = allocateWorkspace();
    if (bcr_status != BCR_SUCCESS) {
        throw std::runtime_error("Failed to allocate workspace");
    }
    
    // Initialize stats
    memset(&stats_, 0, sizeof(BCRStats));
}

CUDABCRSolver::~CUDABCRSolver() {
    freeWorkspace();
    if (cublas_handle_) {
        cublasDestroy(cublas_handle_);
    }
}

BCRStatus CUDABCRSolver::allocateWorkspace() {
    int T = config_.horizon;
    int nx = config_.state_dim;
    int nu = config_.control_dim;
    int n = nx + nu;  // Combined state-control dimension
    
    size_t block_matrix_size = n * n * sizeof(double);
    size_t block_vector_size = n * sizeof(double);
    
    // Calculate total storage needed for all BCR stages
    int max_stages = static_cast<int>(std::ceil(std::log2(T + 1)));
    int total_blocks = 0;
    for (int s = 0; s <= max_stages; ++s) {
        total_blocks += ((T + 1) + (1 << s) - 1) / (1 << s);
    }
    
    // Allocate working memory
    cudaError_t err;
    err = cudaMalloc(&d_work_Q_, total_blocks * block_matrix_size);
    if (err != cudaSuccess) return BCR_ERROR_MEMORY;
    
    err = cudaMalloc(&d_work_B_, total_blocks * block_matrix_size);
    if (err != cudaSuccess) return BCR_ERROR_MEMORY;
    
    err = cudaMalloc(&d_work_q_, total_blocks * block_vector_size);
    if (err != cudaSuccess) return BCR_ERROR_MEMORY;
    
    // Temporary storage
    err = cudaMalloc(&d_temp_Q_, (T + 1) * block_matrix_size);
    if (err != cudaSuccess) return BCR_ERROR_MEMORY;
    
    err = cudaMalloc(&d_temp_B_, T * block_matrix_size);
    if (err != cudaSuccess) return BCR_ERROR_MEMORY;
    
    err = cudaMalloc(&d_temp_q_, (T + 1) * block_vector_size);
    if (err != cudaSuccess) return BCR_ERROR_MEMORY;
    
    // Riccati workspace
    if (config_.use_riccati) {
        err = cudaMalloc(&d_P_, (T + 1) * nx * nx * sizeof(double));
        if (err != cudaSuccess) return BCR_ERROR_MEMORY;
        
        err = cudaMalloc(&d_p_, (T + 1) * nx * sizeof(double));
        if (err != cudaSuccess) return BCR_ERROR_MEMORY;
        
        err = cudaMalloc(&d_K_, T * nu * nx * sizeof(double));
        if (err != cudaSuccess) return BCR_ERROR_MEMORY;
        
        err = cudaMalloc(&d_k_, T * nu * sizeof(double));
        if (err != cudaSuccess) return BCR_ERROR_MEMORY;
    }
    
    return BCR_SUCCESS;
}

void CUDABCRSolver::freeWorkspace() {
    cudaFree(d_work_Q_);
    cudaFree(d_work_B_);
    cudaFree(d_work_q_);
    cudaFree(d_temp_Q_);
    cudaFree(d_temp_B_);
    cudaFree(d_temp_q_);
    
    if (config_.use_riccati) {
        cudaFree(d_P_);
        cudaFree(d_p_);
        cudaFree(d_K_);
        cudaFree(d_k_);
    }
}

BCRStatus CUDABCRSolver::solve(
    const double* d_Q,
    const double* d_B,
    const double* d_q,
    double* d_x
) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int T = config_.horizon;
    int n = config_.state_dim + config_.control_dim;
    
    // Copy input to working arrays
    cudaEventRecord(start);
    
    size_t block_matrix_bytes = n * n * sizeof(double);
    size_t block_vector_bytes = n * sizeof(double);
    
    cudaMemcpy(d_work_Q_, d_Q, (T + 1) * block_matrix_bytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_work_B_, d_B, T * block_matrix_bytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_work_q_, d_q, (T + 1) * block_vector_bytes, cudaMemcpyDeviceToDevice);
    
    // Forward reduction
    BCRStatus status = forwardReduction();
    if (status != BCR_SUCCESS) return status;
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float forward_time;
    cudaEventElapsedTime(&forward_time, start, stop);
    stats_.forward_time_ms = forward_time;
    
    cudaEventRecord(start);
    
    // Backward substitution
    status = backwardSubstitution();
    if (status != BCR_SUCCESS) return status;
    
    // Copy solution
    cudaMemcpy(d_x, d_work_q_, (T + 1) * block_vector_bytes, cudaMemcpyDeviceToDevice);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float backward_time;
    cudaEventElapsedTime(&backward_time, start, stop);
    stats_.backward_time_ms = backward_time;
    stats_.total_time_ms = forward_time + backward_time;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return BCR_SUCCESS;
}

BCRStatus CUDABCRSolver::solveHost(
    const double* h_Q,
    const double* h_B,
    const double* h_q,
    double* h_x
) {
    int T = config_.horizon;
    int n = config_.state_dim + config_.control_dim;
    
    // Allocate device memory
    double *d_Q, *d_B, *d_q, *d_x;
    
    size_t q_size = (T + 1) * n * n * sizeof(double);
    size_t b_size = T * n * n * sizeof(double);
    size_t vec_size = (T + 1) * n * sizeof(double);
    
    cudaMalloc(&d_Q, q_size);
    cudaMalloc(&d_B, b_size);
    cudaMalloc(&d_q, vec_size);
    cudaMalloc(&d_x, vec_size);
    
    // Copy to device
    cudaMemcpy(d_Q, h_Q, q_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, b_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, h_q, vec_size, cudaMemcpyHostToDevice);
    
    // Solve
    BCRStatus status = solve(d_Q, d_B, d_q, d_x);
    
    // Copy back
    cudaMemcpy(h_x, d_x, vec_size, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_Q);
    cudaFree(d_B);
    cudaFree(d_q);
    cudaFree(d_x);
    
    return status;
}

BCRStatus CUDABCRSolver::forwardReduction() {
    int n_blocks = config_.horizon + 1;
    int block_size = config_.state_dim + config_.control_dim;
    
    size_t block_matrix_size = block_size * block_size;
    size_t block_vector_size = block_size;
    
    size_t diag_offset = 0;
    size_t upper_offset = 0;
    size_t rhs_offset = 0;
    
    for (int stage = 0; stage < config_.max_iter; ++stage) {
        int stride = 1 << stage;
        int n_active = (n_blocks + stride - 1) / stride;
        int n_reduced = (n_active + 1) / 2;
        
        if (n_active <= 1) break;
        
        // Calculate next offsets
        size_t next_diag = diag_offset + n_active * block_matrix_size;
        size_t next_upper = upper_offset + n_active * block_matrix_size;
        size_t next_rhs = rhs_offset + n_active * block_vector_size;
        
        // Launch kernel
        int threads = 256;
        int blocks = n_reduced;
        size_t smem_size = (7 * block_size * block_size + 2 * block_size) * sizeof(double);
        
        bcrForwardReduction<<<blocks, threads, smem_size>>>(
            d_work_Q_ + diag_offset,
            d_work_B_ + upper_offset,
            d_work_q_ + rhs_offset,
            d_work_Q_ + next_diag,
            d_work_B_ + next_upper,
            d_work_q_ + next_rhs,
            block_size,
            stride,
            n_active
        );
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return BCR_ERROR_CUDA;
        
        // Update
        diag_offset = next_diag;
        upper_offset = next_upper;
        rhs_offset = next_rhs;
        n_blocks = n_reduced;
        
        stats_.num_stages = stage + 1;
    }
    
    return BCR_SUCCESS;
}

BCRStatus CUDABCRSolver::backwardSubstitution() {
    int block_size = config_.state_dim + config_.control_dim;
    
    // Work backwards through stages
    for (int stage = stats_.num_stages - 1; stage >= 0; --stage) {
        int stride = 1 << stage;
        int n_blocks_to_solve = config_.horizon / stride;
        
        int threads = 256;
        int blocks = n_blocks_to_solve;
        size_t smem_size = (block_size * block_size + 2 * block_size) * sizeof(double);
        
        bcrBackwardSubstitution<<<blocks, threads, smem_size>>>(
            d_work_Q_,
            d_work_B_,
            d_work_q_,
            d_work_q_,  
            block_size,
            stride,
            n_blocks_to_solve
        );
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return BCR_ERROR_CUDA;
    }
    
    return BCR_SUCCESS;
}

// C API implementations
extern "C" {

void* bcr_create(const BCRConfig* config) {
    try {
        return new CUDABCRSolver(*config);
    } catch (...) {
        return nullptr;
    }
}

void bcr_destroy(void* solver) {
    delete static_cast<CUDABCRSolver*>(solver);
}

BCRStatus bcr_solve(void* solver, const double* Q, const double* B,
                    const double* q, double* x) {
    return static_cast<CUDABCRSolver*>(solver)->solveHost(Q, B, q, x);
}

BCRStats bcr_get_stats(void* solver) {
    return static_cast<CUDABCRSolver*>(solver)->getStats();
}

} // extern "C"