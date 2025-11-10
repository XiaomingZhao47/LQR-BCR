# tests for Block Cyclic Reduction solver

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bcr.solver import HermitianCyclicReduction
from bcr.systems import BlockTridiagonalSystem


def test_bcr_diagonal():
    """test BCR on diagonal system"""
    N = 4
    m = 2
    
    A_blocks = [np.eye(m) * (i + 1) for i in range(N)]
    B_blocks = [np.zeros((m, m)) for _ in range(N-1)]
    b_blocks = [np.ones(m) for _ in range(N)]
    
    system = BlockTridiagonalSystem(B_blocks, A_blocks, b_blocks)
    
    solver = HermitianCyclicReduction()
    solution, stats = solver.solve(system)
    
    for i, x in enumerate(solution):
        expected = np.ones(m) / (i + 1)
        assert np.allclose(x, expected, atol=1e-10)
    
    print(f"BCR diagonal test passed (time: {stats['time']:.4f}s)")


def test_bcr_tridiagonal():
    """test BCR on tridiagonal system"""
    N = 8
    m = 1
    
    A_blocks = [np.array([[2.0]]) for _ in range(N)]
    B_blocks = [np.array([[-1.0]]) for _ in range(N-1)]
    b_blocks = [np.array([1.0]) for _ in range(N)]
    
    system = BlockTridiagonalSystem(B_blocks, A_blocks, b_blocks)
    
    assert system.verify_hermitian()
    is_pd, min_eig, max_eig = system.verify_positive_definite()
    assert is_pd
    
    solver = HermitianCyclicReduction(verbose=False)
    solution, stats = solver.solve(system)
    
    A_dense, b_dense = system.to_dense()
    x_flat = np.concatenate(solution)
    residual = A_dense @ x_flat - b_dense
    
    assert np.linalg.norm(residual) < 1e-10
    
    print(f"BCR tridiagonal test passed (levels: {stats['levels']})")


def test_bcr_power_of_two():
    """test BCR requires power of 2"""
    N = 7  # not power of 2
    m = 2
    
    A_blocks = [np.eye(m) * 2 for _ in range(N)]
    B_blocks = [np.zeros((m, m)) for _ in range(N-1)]
    b_blocks = [np.ones(m) for _ in range(N)]
    
    system = BlockTridiagonalSystem(B_blocks, A_blocks, b_blocks)
    
    solver = HermitianCyclicReduction()
    solution, stats = solver.solve(system)
    
    assert len(solution) == N 
    
    print(f"BCR padding test passed (N={N} -> padded to 8)")


if __name__ == '__main__':
    test_bcr_diagonal()
    test_bcr_tridiagonal()
    test_bcr_power_of_two()
    print("\\nAll BCR tests passed!")