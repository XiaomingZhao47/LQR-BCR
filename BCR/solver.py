# Hermitian Positive Definite Block Cyclic Reduction Solver

"""
Neuenhofen, M. (2018). "Review of Cyclic Reduction for Parallel Solution 
of Hermitian Positive Definite Block-Tridiagonal Linear Systems"
"""

import numpy as np
import scipy.linalg as la
import time
from typing import Tuple, List

from .systems import BlockTridiagonalSystem


class HermitianCyclicReduction:
    """
    solve block-tridiagonal HPD systems using cyclic reduction
    
    complexity
        - time O(N log N)
        - depth O(log N)
        - memory: O(N)
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.stats = {}
    
    def solve(self, system: BlockTridiagonalSystem) -> Tuple[List[np.ndarray], dict]:
        """
        Solve the block-tridiagonal system
        
        Args:
            system: Block-tridiagonal HPD system
            
        Returns:
            solution: List of solution blocks
            stats: Dictionary with solve statistics
        """
        start_time = time.time()
        
        # verify system properties
        if not system.verify_hermitian():
            raise ValueError("System must be Hermitian!")
        
        is_pd, min_eig, max_eig = system.verify_positive_definite()
        if not is_pd:
            raise ValueError(f"Not positive definite! Min eig: {min_eig:.6e}")
        
        if self.verbose:
            print(f"HPD system: eigs [{min_eig:.6f}, {max_eig:.6f}], "
                  f"cond={max_eig/min_eig:.2f}")
        
        N = system.N
        original_N = N
        
        if not self._is_power_of_two(N):
            system = self._pad_system(system)
        
        solution = self._solve_recursive(system)
        
        if original_N != system.N:
            solution = solution[:original_N]
        
        self.stats = {
            'time': time.time() - start_time,
            'N': original_N,
            'block_size': system.block_size,
            'levels': self.stats.get('levels', 0)
        }
        
        return solution, self.stats
    
    def _is_power_of_two(self, n: int) -> bool:
        """Check if n is a power of 2"""
        return n > 0 and (n & (n - 1)) == 0
    
    def _pad_system(self, system: BlockTridiagonalSystem) -> BlockTridiagonalSystem:
        """Pad system to next power of 2"""
        N = system.N
        next_pow2 = 2 ** int(np.ceil(np.log2(N)))
        n_pad = next_pow2 - N
        
        if n_pad == 0:
            return system
        
        m = system.block_size
        
        B_padded = list(system.B_blocks) + [np.zeros((m, m)) for _ in range(n_pad)]
        A_padded = list(system.A_blocks) + [np.eye(m) * 1e6 for _ in range(n_pad)]
        b_padded = list(system.b_blocks) + [np.zeros(m) for _ in range(n_pad)]
        
        return BlockTridiagonalSystem(B_padded, A_padded, b_padded)
    
    def _solve_recursive(self, system: BlockTridiagonalSystem) -> List[np.ndarray]:
        """Recursive cyclic reduction"""
        N = system.N
        
        # single block
        if N == 1:
            L = la.cholesky(system.A_blocks[0], lower=True)
            z = la.solve_triangular(L, system.b_blocks[0], lower=True)
            return [la.solve_triangular(L.conj().T, z, lower=False)]
        
        # separate and recursive solve subproblem
        system_odd, system_even = self._reduction_step(system)
        
        if self.verbose:
            print(f"Level: N={N} -> odd={system_odd.N}, even={system_even.N}")
        
        x_odd = self._solve_recursive(system_odd)
        x_even = self._solve_recursive(system_even)
        
        solution = [None] * N
        for j in range(len(x_odd)):
            solution[2*j] = x_odd[j]
        for j in range(len(x_even)):
            solution[2*j + 1] = x_even[j]
        
        self.stats['levels'] = self.stats.get('levels', 0) + 1
        return solution
    
    def _reduction_step(self, system: BlockTridiagonalSystem) -> Tuple[BlockTridiagonalSystem, BlockTridiagonalSystem]:
        """
        cyclic reduction step from Neuenhofen Eq
        
        eliminates odd-indexed blocks, creating two smaller systems
        """
        N = system.N
        N_half = N // 2
        B, A, y = system.B_blocks, system.A_blocks, system.b_blocks
        
        U, E, u = [], [], []  # odd 
        V, F, v = [], [], []  # even 
        
        for j in range(N_half):
            idx = 2 * j
            
            if idx > 0:
                B_l, A_l, y_l = B[idx-1], A[idx-1], y[idx-1]
            else:
                m = A[0].shape[0]
                B_l, A_l, y_l = np.zeros((m,m)), np.eye(m), np.zeros(m)
            
            A_c, y_c = A[idx], y[idx]
            
            if idx + 1 < N:
                B_r, A_r, y_r = B[idx], A[idx+1], y[idx+1]
            else:
                m = A[0].shape[0]
                B_r, A_r, y_r = np.zeros((m,m)), np.eye(m), np.zeros(m)
            
            # Schur complement 
            U_j = A_c.copy()
            if idx > 0:
                U_j -= self._schur(A_l, B_l)
            if idx + 1 < N:
                U_j -= self._schur(A_r, B_r).conj().T
            
            u_j = y_c.copy()
            if idx > 0:
                u_j -= B_l @ self._solve(A_l, y_l)
            if idx + 1 < N:
                u_j -= B_r.conj().T @ self._solve(A_r, y_r)
            
            U.append(U_j)
            u.append(u_j)
            
            if j < N_half - 1 and 2*j+1 < N and 2*j+2 < N:
                E.append(-B[2*j+1] @ self._solve(A[2*j+1], B[2*j]))
        
        for j in range(N_half):
            idx = 2*j + 1
            if idx >= N:
                break
            
            if idx > 0:
                B_l, A_l, y_l = B[idx-1], A[idx-1], y[idx-1]
            else:
                m = A[0].shape[0]
                B_l, A_l, y_l = np.zeros((m,m)), np.eye(m), np.zeros(m)
            
            A_c, y_c = A[idx], y[idx]
            
            if idx + 1 < N:
                B_r, A_r, y_r = B[idx], A[idx+1], y[idx+1]
            else:
                m = A[0].shape[0]
                B_r, A_r, y_r = np.zeros((m,m)), np.eye(m), np.zeros(m)
            
            V_j = A_c.copy()
            if idx > 0:
                V_j -= self._schur(A_l, B_l)
            if idx + 1 < N:
                V_j -= self._schur(A_r, B_r).conj().T
            
            v_j = y_c.copy()
            if idx > 0:
                v_j -= B_l @ self._solve(A_l, y_l)
            if idx + 1 < N:
                v_j -= B_r.conj().T @ self._solve(A_r, y_r)
            
            V.append(V_j)
            v.append(v_j)
            
            if j < N_half - 1 and 2*j+2 < N and 2*j+3 < N:
                F.append(-B[2*j+2] @ self._solve(A[2*j+2], B[2*j+1]))
        
        return (BlockTridiagonalSystem(E, U, u),
                BlockTridiagonalSystem(F, V, v))
    
    def _solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """solve Ax = b by cholesky"""
        L = la.cholesky(A, lower=True)
        z = la.solve_triangular(L, b, lower=True)
        return la.solve_triangular(L.conj().T, z, lower=False)
    
    def _schur(self, M: np.ndarray, G: np.ndarray) -> np.ndarray:
        """schur complement"""
        L = la.cholesky(M, lower=True)
        G_tilde = la.solve_triangular(L, G, lower=True)
        return G_tilde.conj().T @ G_tilde