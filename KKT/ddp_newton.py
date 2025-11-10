# DDP Newton system builder
# block-tridiagonal HPD systems for DDP/iLQR iterations

import numpy as np
from typing import List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bcr.systems import BlockTridiagonalSystem


class DDPNewtonSystem:
    """
    Newton system for DDP/iLQR iterations
    
    as optimal trajectory, creates a block-tridiagonal
    HPD system suitable for cyclic reduction
    """
    
    def __init__(self, A, B, Q, R, Q_f, N):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.Q_f = Q_f
        self.N = N
        
        self.n = A.shape[0]
        self.m = B.shape[1]
    
    def build_newton_system(self, x_nom: List[np.ndarray], u_nom: List[np.ndarray],
                           P: List[np.ndarray]) -> BlockTridiagonalSystem:
        """
        build Newton system around nominal trajectory
        
        args:
            x_nom: nominal state trajectory
            u_nom: nominal control trajectory
            P: cost matrices from Riccati
            
        returns:
            block-tridiagonal HPD system for control perturbations
        """
        N, n, m = self.N, self.n, self.m
        
        A_blocks = []
        B_blocks = []
        b_blocks = []
        
        for k in range(N):
            H_k = self.R + self.B.T @ P[k+1] @ self.B
            A_blocks.append(H_k)
            
            if k < N - 1:
                C_k = np.zeros((m, m))
                B_blocks.append(C_k)
            
            # RHS
            g_k = np.zeros(m)
            b_blocks.append(g_k)
        
        return BlockTridiagonalSystem(B_blocks, A_blocks, b_blocks)
