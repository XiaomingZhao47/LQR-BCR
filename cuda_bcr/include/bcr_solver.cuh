#ifndef BCR_SOLVER_CUH
#define BCR_SOLVER_CUH

#include "types.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

/**
 * BCR solver for LQR 
 * 
 * solves block-tridiagonal KKT system 
 * [Q0  B0^T              ] [x0]   [q0]
 * [B0  Q1    B1^T        ] [x1]   [q1]
 * [    B1    Q2   B2^T   ] [x2] = [q2]
 * [          ...  ...  ...] [..]   [..]
 * [              BT-1  QT ] [xT]   [qT]
 */

class CUDABCRSolver {
public:
    /**
     * constructor
     * @param config solver configuration
     */
    explicit CUDABCRSolver(const BCRConfig& config);
    
    /**
     * destructor - cleans up CUDA resources
     */
    ~CUDABCRSolver();
    
    /**
     * solve LQR problem using BCR
     * 
     * @param d_Q diagonal blocks         [T+1, nx+nu, nx+nu] (device)
     * @param d_B off-diagonal blocks     [T, nx, nx+nu]      (device)
     * @param d_q RHS vectors             [T+1, nx+nu]        (device)
     * @param d_x solution                [T+1, nx+nu]        (output, device)
     * @return    status code
     */
    BCRStatus solve(
        const double* d_Q,
        const double* d_B,
        const double* d_q,
        double* d_x
    );
    
    /**
     * solve with host memory 
     */
    BCRStatus solveHost(
        const double* h_Q,
        const double* h_B,
        const double* h_q,
        double* h_x
    );
    
    /**
     * Get statistics from last solve
     */
    BCRStats getStats() const { return stats_; }
    
    /**
     * Compare with Riccati recursion (for validation)
     */
    BCRStatus solveRiccati(
        const double* d_Q,
        const double* d_B,
        const double* d_q,
        double* d_x
    );

private:
    BCRConfig config_;
    BCRStats stats_;
    cublasHandle_t cublas_handle_;
    
    // Device memory workspaces
    double* d_work_Q_;      // Working diagonal blocks
    double* d_work_B_;      // Working off-diagonal blocks
    double* d_work_q_;      // Working RHS
    double* d_temp_Q_;      // Temporary storage
    double* d_temp_B_;
    double* d_temp_q_;
    
    // Riccati recursion workspace
    double* d_P_;           // Value function Hessian
    double* d_p_;           // Value function gradient
    double* d_K_;           // Feedback gains
    double* d_k_;           // Feedforward terms
    
    // Methods
    BCRStatus allocateWorkspace();
    void freeWorkspace();
    BCRStatus forwardReduction();
    BCRStatus backwardSubstitution();
    BCRStatus riccatiBackward(const double* d_Q, const double* d_B, const double* d_q);
    BCRStatus riccatiForward(const double* d_K, const double* d_k, double* d_x);
};

// C API for Python bindings
extern "C" {
    void* bcr_create(const BCRConfig* config);
    void bcr_destroy(void* solver);
    BCRStatus bcr_solve(void* solver, const double* Q, const double* B, 
                        const double* q, double* x);
    BCRStats bcr_get_stats(void* solver);
}

#endif // BCR_SOLVER_CUH