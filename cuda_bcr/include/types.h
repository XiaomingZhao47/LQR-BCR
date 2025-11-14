#ifndef CUDA_BCR_TYPES_H
#define CUDA_BCR_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

// status codes
typedef enum {
    BCR_SUCCESS = 0,
    BCR_ERROR_INVALID_ARGS = -1,
    BCR_ERROR_CUDA = -2,
    BCR_ERROR_MEMORY = -3,
    BCR_ERROR_NOT_CONVERGED = -4
} BCRStatus;

// solver configuration
typedef struct {
    int horizon;           // time horizon T
    int state_dim;         // state dimension nx
    int control_dim;       // control dimension nu
    int max_iter;          // maximum BCR iterations
    double tolerance;      // convergence tolerance
    bool use_riccati;      // Riccati recursion comparison
    bool verbose;          // debug info
} BCRConfig;

// solver stats
typedef struct {
    int num_stages;        // number of BCR stages
    double forward_time_ms;
    double backward_time_ms;
    double total_time_ms;
    int cuda_blocks_used;
    int threads_per_block;
} BCRStats;

#ifdef __cplusplus
}
#endif

#endif // CUDA_BCR_TYPES_H