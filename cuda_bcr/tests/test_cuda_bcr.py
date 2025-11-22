import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from cuda_bcr import BCRSolver, is_cuda_available
from cuda_bcr.utils import format_lqr_system, validate_solution

pytestmark = pytest.mark.skipif(
    not is_cuda_available(),
    reason="CUDA BCR module not available"
)

def test_basic_solve():
    """Test basic BCR solve."""
    horizon = 10
    nx, nu = 4, 2
    n = nx + nu
    
    # Generate SPD system
    np.random.seed(42)
    Q = np.random.randn(horizon + 1, n, n)
    Q = Q @ Q.transpose(0, 2, 1) + np.eye(n)
    
    B = np.random.randn(horizon, nx, n) * 0.1
    q = np.random.randn(horizon + 1, n)
    
    # Solve
    solver = BCRSolver(horizon, nx, nu)
    x = solver.solve(Q, B, q)
    
    assert x.shape == (horizon + 1, n)
    assert np.all(np.isfinite(x))


def test_lqr_format():
    """Test solving from LQR matrices."""
    T = 20
    nx, nu = 6, 3
    
    # Random LQR problem
    np.random.seed(123)
    Q = np.array([np.eye(nx) for _ in range(T + 1)])
    R = np.array([np.eye(nu) for _ in range(T)])
    A = np.array([np.eye(nx) + np.random.randn(nx, nx) * 0.1 for _ in range(T)])
    B = np.array([np.random.randn(nx, nu) for _ in range(T)])
    q = np.zeros((T + 1, nx))
    r = np.zeros((T, nu))
    x0 = np.random.randn(nx)
    
    # Solve
    solver = BCRSolver(T, nx, nu)
    x_traj, u_traj = solver.solve_from_lqr_matrices(Q, R, A, B, q, r, x0)
    
    # Validate
    is_valid, max_viol = validate_solution(x_traj, u_traj, A, B, x0)
    print(f"Dynamics violation: {max_viol:.2e}")
    
    assert is_valid, f"Dynamics violated by {max_viol}"


def test_vs_cpu():
    """Compare CUDA BCR with CPU Riccati."""
    from LQR.solvers import RiccatiSolver
    
    horizon = 50
    nx, nu = 8, 4
    n = nx + nu
    
    # Generate problem
    np.random.seed(999)
    Q_kkt = np.random.randn(horizon + 1, n, n)
    Q_kkt = Q_kkt @ Q_kkt.transpose(0, 2, 1) + np.eye(n)
    B_kkt = np.random.randn(horizon, nx, n) * 0.1
    q_kkt = np.random.randn(horizon + 1, n)
    
    # CPU solve
    cpu_solver = RiccatiSolver(horizon, nx, nu)
    x_cpu = cpu_solver.solve(Q_kkt, B_kkt, q_kkt)
    
    # CUDA solve
    cuda_solver = BCRSolver(horizon, nx, nu)
    x_cuda = cuda_solver.solve(Q_kkt, B_kkt, q_kkt)
    
    # Compare
    rel_error = np.linalg.norm(x_cuda - x_cpu) / np.linalg.norm(x_cpu)
    print(f"Relative error: {rel_error:.2e}")
    
    assert rel_error < 1e-4, f"Solutions differ by {rel_error}"
    
    # Print stats
    stats = cuda_solver.get_stats()
    print(f"BCR stages: {stats['num_stages']}")
    print(f"BCR time: {stats['total_time_ms']:.3f} ms")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])