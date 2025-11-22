# test_thomas.py
import numpy as np
import time
from LQR.solvers import RiccatiSolver, ThomasBlockSolver

def test_block_tridiagonal():
    """Test Thomas solver on a simple block-tridiagonal system."""
    print("="*70)
    print("TEST 1: Block-Tridiagonal System")
    print("="*70)
    
    N = 10
    block_size = 3
    
    np.random.seed(42)
    
    # random SPD diagonal blocks
    diagonal = np.zeros((N + 1, block_size, block_size))
    for k in range(N + 1):
        temp = np.random.randn(block_size, block_size)
        diagonal[k] = temp @ temp.T + np.eye(block_size)
    
    # random upper blocks
    upper = np.random.randn(N, block_size, block_size) * 0.1
    
    # random RHS
    rhs = np.random.randn(N + 1, block_size)
    
    # solve with Thomas
    thomas = ThomasBlockSolver(N, block_size)
    solution = thomas.solve(diagonal, upper, rhs)
    
    # computing residual: ||A*x - b||
    residual = 0.0
    for k in range(N + 1):
        res_k = diagonal[k] @ solution[k] - rhs[k]
        
        if k > 0:
            res_k += upper[k-1].T @ solution[k-1]
        if k < N:
            res_k += upper[k] @ solution[k+1]
        
        residual += np.linalg.norm(res_k)**2
    
    residual = np.sqrt(residual)
    
    print(f"Block size: {block_size}")
    print(f"Horizon: {N}")
    print(f"Residual: {residual:.2e}")
    print(f"Result: {'PASS' if residual < 1e-10 else 'FAIL'}")
    print()


def test_lqr_comparison():
    """Compare Riccati and Thomas solvers - but note the limitation."""
    print("="*70)
    print("TEST 2: Riccati Solver for LQR")
    print("="*70)
    
    # LQR problem
    n, m = 4, 2
    N = 20
    
    np.random.seed(123)
    A = np.eye(n) + np.random.randn(n, n) * 0.01
    B = np.random.randn(n, m) * 0.1
    Q = np.eye(n) * 10
    R = np.eye(m)
    Q_f = np.eye(n) * 10
    
    x_init = np.random.randn(n)
    
    print(f"State dim: {n}, Control dim: {m}, Horizon: {N}")
    print()
    
    # Solve with Riccati
    print("Riccati Solver:")
    riccati = RiccatiSolver(A, B, Q, R, Q_f, N)
    
    start = time.perf_counter()
    states_r, controls_r, cost_r = riccati.solve(x_init)
    riccati_time = (time.perf_counter() - start) * 1000
    
    print(f"   Time: {riccati_time:.3f} ms")
    print(f"   Cost: {cost_r:.6f}")
    print()
    
    print("Note: Thomas solver is for generic block-tridiagonal systems.")
    print("      For LQR problems, use Riccati solver (which is faster).")
    print("      For benchmarking vs BCR, generate block-tridiagonal test systems.")
    print()

def test_scaling():
    """Test scaling of Thomas solver on generic block-tridiagonal systems."""
    print("="*70)
    print("TEST 3: Thomas Solver Performance Scaling")
    print("="*70)
    
    block_sizes = [6, 12, 20]
    horizons = [10, 20, 50, 100]
    
    print(f"{'Block Size':>12} {'Horizon':>10} {'Time (ms)':>15} {'Time/N (ms)':>15}")
    print("-"*70)
    
    for block_size in block_sizes:
        for N in horizons:
            np.random.seed(42)
            
            # Generate random SPD system
            diagonal = np.zeros((N + 1, block_size, block_size))
            for k in range(N + 1):
                temp = np.random.randn(block_size, block_size)
                diagonal[k] = temp @ temp.T + np.eye(block_size)
            
            upper = np.random.randn(N, block_size, block_size) * 0.1
            rhs = np.random.randn(N + 1, block_size)
            
            # Solve with Thomas
            thomas = ThomasBlockSolver(N, block_size)
            
            times = []
            for _ in range(5):
                start = time.perf_counter()
                solution = thomas.solve(diagonal, upper, rhs)
                times.append((time.perf_counter() - start) * 1000)
            
            thomas_time = np.mean(times)
            time_per_n = thomas_time / N
            
            print(f"{block_size:>12} {N:>10} {thomas_time:>15.3f} {time_per_n:>15.4f}")
    
    print()
    print("Expected: Time should scale as O(N * block_size^3)")
    print()


if __name__ == '__main__':
    test_block_tridiagonal()
    test_lqr_comparison()
    test_scaling()
    
    print("="*70)
    print("All tests complete!")
    print("="*70)