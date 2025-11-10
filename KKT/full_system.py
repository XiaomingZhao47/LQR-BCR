# KKT system construction with Lagrange multipliers

import numpy as np
from typing import List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bcr.systems import BlockTridiagonalSystem


class FullKKTSystem:
    """
    full KKT system for LQR with Lagrange multipliers
    
    variables per stage k: [u_k; x_{k+1}; lambda_{k+1}]
    - u_k: control (m-dimensional)
    - x_{k+1}: next state (n-dimensional)
    - lambda_{k+1}: Lagrange multiplier for dynamics (n-dimensional)
    
    Creates block-tridiagonal structure (but indefinite/saddle-point)
    """
    
    def __init__(self, A_list: List[np.ndarray], B_list: List[np.ndarray],
                 Q_list: List[np.ndarray], R_list: List[np.ndarray], x_init: np.ndarray):
        """
        Args:
            A_list: State transition matrices [A_0, ..., A_{N-1}]
            B_list: Control matrices [B_0, ..., B_{N-1}]
            Q_list: State cost matrices [Q_0, ..., Q_N] (includes Q_f)
            R_list: Control cost matrices [R_0, ..., R_{N-1}]
            x_init: Initial state
        """
        self.A_list = A_list
        self.B_list = B_list
        self.Q_list = Q_list
        self.R_list = R_list
        self.x_init = x_init
        
        self.N = len(A_list)
        self.n = A_list[0].shape[0]
        self.m = B_list[0].shape[1]
    
    def build_full_kkt(self) -> BlockTridiagonalSystem:
        """
        Build full KKT system with block-tridiagonal structure
        
        KKT conditions at stage k
          ∂ as P, partial derivative
          PL/Pu_k = R_k u_k + B_k^T lambda_{k+1} = 0
          PL/Px_{k+1} = Q_{k+1} x_{k+1} - lambda_{k+1} + A_{k+1}^T lambda_{k+2} = 0
          PL/Plambda_{k+1} = A_k x_k + B_k u_k - x_{k+1} = 0
        
        return block-tridiagonal system
        """
        N, n, m = self.N, self.n, self.m
        
        A_blocks = []
        B_blocks = []
        b_blocks = []
        
        block_size = m + 2 * n
        
        for k in range(N):
            A_k = self.A_list[k]
            B_k = self.B_list[k]
            Q_k1 = self.Q_list[k + 1]
            R_k = self.R_list[k]
            
            # diagonal block for [u_k; x_{k+1}; lambda_{k+1}]
            diag_block = np.zeros((block_size, block_size))
            
            # row 1 PL/Pu_k
            diag_block[0:m, 0:m] = R_k
            diag_block[0:m, m+n:m+2*n] = B_k.T
            
            # row 2 PL/Px_{k+1}
            diag_block[m:m+n, m:m+n] = Q_k1
            diag_block[m:m+n, m+n:m+2*n] = -np.eye(n)
            
            # row 3 PL/P_lambda_{k+1} 
            diag_block[m+n:m+2*n, 0:m] = B_k
            diag_block[m+n:m+2*n, m:m+n] = -np.eye(n)
            
            A_blocks.append(diag_block)
            
            if k < N - 1:
                A_k1 = self.A_list[k + 1]
                off_block = np.zeros((block_size, block_size))
                
                off_block[m:m+n, m+n:m+2*n] = A_k1.T
                
                B_blocks.append(off_block)
            
            # RHS 
            rhs = np.zeros(block_size)
            if k == 0:
                rhs[m+n:m+2*n] = -A_k @ self.x_init
            
            b_blocks.append(rhs)
        
        return BlockTridiagonalSystem(B_blocks, A_blocks, b_blocks)