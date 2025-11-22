#!/usr/bin/env python3
"""
Demo of CUDA BCR solver.
"""

import numpy as np
import time
from cuda_bcr import BCRSolver
from cuda_bcr.utils import format_lqr_system, validate_solution


def main():
    print("CUDA BCR Demo")
    print("=" * 60)
    
    # Problem setup
    T = 100
    nx, nu = 10, 5
    
    # Random LQR problem
    np.random.seed(42)
    Q = np.array([np.eye(nx) * 10 for _ in range(T + 1)])
    R = np.array([np.eye(nu) for _ in range(T)])
    A = np.array([np.eye(nx) + np.random.randn(nx, nx) * 0.05 for _ in range(T)])
    B = np.array([np.random.randn(nx, nu) * 0.1 for _ in range(T)])
    
    q = np.zeros((T + 1, nx))
    r = np.zeros((T, nu))
    x0 = np.random.randn(nx)
    
    print(f"Horizon: {T}")
    print(f"State dim: {nx}, Control dim: {nu}")
    print()
    
    # Create solver
    solver = BCRSolver(T, nx, nu, verbose=True)
    
    # Solve
    print("Solving...")
    start = time.perf_counter()
    x_traj, u_traj = solver.solve_from_lqr_matrices(Q, R, A, B, q, r, x0)
    elapsed = time.perf_counter() - start
    
    print(f"Solved in {elapsed*1000:.2f} ms")
    print()
    
    # Get statistics
    stats = solver.get_stats()
    print("Solver Statistics:")
    print(f"  BCR stages: {stats['num_stages']}")
    print(f"  Forward pass: {stats['forward_time_ms']:.3f} ms")
    print(f"  Backward pass: {stats['backward_time_ms']:.3f} ms")
    print(f"  Total: {stats['total_time_ms']:.3f} ms")
    print()
    
    # Validate
    is_valid, max_viol = validate_solution(x_traj, u_traj, A, B, x0)
    print(f"Dynamics validation: {'PASS' if is_valid else 'FAIL'}")
    print(f"Max violation: {max_viol:.2e}")
    
    # Compute cost
    cost = 0.0
    for k in range(T):
        cost += x_traj[k] @ Q[k] @ x_traj[k] + u_traj[k] @ R[k] @ u_traj[k]
    cost += x_traj[T] @ Q[T] @ x_traj[T]
    print(f"Total cost: {cost:.6f}")


if __name__ == '__main__':
    main()