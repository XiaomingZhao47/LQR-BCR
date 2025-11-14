import numpy as np
import time
from typing import Tuple, Dict, Optional
try:
    from _cuda_bcr import BCRSolver as _BCRSolver
except ImportError:
    _BCRSolver = None
    print("Warning: CUDA BCR module not found. Install with 'pip install -e .'")


class BCRSolver:
    """
    CUDA Block Cyclic Reduction solver for LQR problems.
    
    Solves the block-tridiagonal KKT system:
    [Q0  B0^T              ] [x0]   [q0]
    [B0  Q1    B1^T        ] [x1]   [q1]
    [    B1    Q2   B2^T   ] [x2] = [q2]
    [          ...  ...  ...] [..]   [..]
    [              BT-1  QT ] [xT]   [qT]
    
    Args:
        horizon: Time horizon T
        state_dim: State dimension nx
        control_dim: Control dimension nu
        max_iter: Maximum number of BCR iterations
        tolerance: Convergence tolerance
        verbose: Print debug information
    """
    
    def __init__(self, horizon: int, state_dim: int, control_dim: int,
                 max_iter: int = 100, tolerance: float = 1e-6,
                 verbose: bool = False):
        if _BCRSolver is None:
            raise RuntimeError("CUDA BCR module not available")
        
        self.horizon = horizon
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.block_size = state_dim + control_dim
        
        self._solver = _BCRSolver(
            horizon, state_dim, control_dim,
            max_iter, tolerance, verbose
        )
    
    def solve(self, Q: np.ndarray, B: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Solve LQR problem.
        
        Args:
            Q: Diagonal blocks [T+1, n, n] where n = nx + nu
            B: Off-diagonal blocks [T, nx, n]
            q: RHS vectors [T+1, n]
        
        Returns:
            x: Solution [T+1, n]
        """
        # Validate inputs
        assert Q.shape == (self.horizon + 1, self.block_size, self.block_size)
        assert B.shape == (self.horizon, self.state_dim, self.block_size)
        assert q.shape == (self.horizon + 1, self.block_size)
        
        # Ensure contiguous and correct dtype
        Q = np.ascontiguousarray(Q, dtype=np.float64)
        B = np.ascontiguousarray(B, dtype=np.float64)
        q = np.ascontiguousarray(q, dtype=np.float64)
        
        # Solve
        x = self._solver.solve(Q, B, q)
        
        return x
    
    def get_stats(self) -> Dict[str, float]:
        """Get solver statistics from last solve."""
        return self._solver.get_stats()
    
    def solve_from_lqr_matrices(
        self,
        Q: np.ndarray,  # [T+1, nx, nx]
        R: np.ndarray,  # [T, nu, nu]
        A: np.ndarray,  # [T, nx, nx]
        B: np.ndarray,  # [T, nx, nu]
        q: np.ndarray,  # [T+1, nx]
        r: np.ndarray,  # [T, nu]
        x0: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve LQR from standard matrices.
        
        Constructs KKT system and solves for state and control trajectories.
        
        Returns:
            x: State trajectory [T+1, nx]
            u: Control trajectory [T, nu]
        """
        from .utils import format_lqr_system
        
        # Format into KKT system
        Q_kkt, B_kkt, q_kkt = format_lqr_system(Q, R, A, B, q, r, x0)
        
        # Solve
        solution = self.solve(Q_kkt, B_kkt, q_kkt)
        
        # Extract state and control
        x_traj = solution[:, :self.state_dim]
        u_traj = solution[:-1, self.state_dim:]
        
        return x_traj, u_traj


def benchmark_vs_cpu(horizon: int, state_dim: int, control_dim: int,
                     n_trials: int = 10) -> Dict[str, float]:
    """
    Benchmark CUDA BCR against CPU Riccati recursion.
    
    Args:
        horizon: Time horizon
        state_dim: State dimension
        control_dim: Control dimension
        n_trials: Number of trials for timing
    
    Returns:
        Dictionary with timing results
    """
    # Import CPU solver
    import sys
    sys.path.insert(0, '..')
    from LQR.solvers import RiccatiSolver
    
    # Generate random problem
    np.random.seed(42)
    n = state_dim + control_dim
    
    Q_kkt = np.random.randn(horizon + 1, n, n)
    Q_kkt = Q_kkt @ Q_kkt.transpose(0, 2, 1) + np.eye(n)  # Make SPD
    
    B_kkt = np.random.randn(horizon, state_dim, n) * 0.1
    q_kkt = np.random.randn(horizon + 1, n)
    
    # CPU solver
    cpu_solver = RiccatiSolver(horizon, state_dim, control_dim)
    
    cpu_times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        x_cpu = cpu_solver.solve(Q_kkt, B_kkt, q_kkt)
        cpu_times.append(time.perf_counter() - start)
    
    # CUDA solver
    cuda_solver = BCRSolver(horizon, state_dim, control_dim)
    
    # Warm-up
    _ = cuda_solver.solve(Q_kkt, B_kkt, q_kkt)
    
    cuda_times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        x_cuda = cuda_solver.solve(Q_kkt, B_kkt, q_kkt)
        cuda_times.append(time.perf_counter() - start)
    
    # Verify correctness
    x_cpu = cpu_solver.solve(Q_kkt, B_kkt, q_kkt)
    x_cuda = cuda_solver.solve(Q_kkt, B_kkt, q_kkt)
    error = np.linalg.norm(x_cuda - x_cpu) / np.linalg.norm(x_cpu)
    
    stats = cuda_solver.get_stats()
    
    return {
        'cpu_mean_ms': np.mean(cpu_times) * 1000,
        'cpu_std_ms': np.std(cpu_times) * 1000,
        'cuda_mean_ms': np.mean(cuda_times) * 1000,
        'cuda_std_ms': np.std(cuda_times) * 1000,
        'speedup': np.mean(cpu_times) / np.mean(cuda_times),
        'relative_error': error,
        'bcr_stages': stats['num_stages'],
        'bcr_forward_ms': stats['forward_time_ms'],
        'bcr_backward_ms': stats['backward_time_ms']
    }