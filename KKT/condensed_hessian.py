# condensed Hessian construction

import numpy as np
from typing import List


class CondensedHessianBuilder:
    """
    build condensed Hessian by eliminating states
    
    variables, only controls [u_0, u_1, ..., u_{N-1}]
    
    the resulting Hessian is DENSE with bandwidth N-1,
    unsuitable for block-tridiagonal solvers
    """
    
    def __init__(self, A_list: List[np.ndarray], B_list: List[np.ndarray],
                 Q_list: List[np.ndarray], R_list: List[np.ndarray], x_init: np.ndarray):
        self.A_list = A_list
        self.B_list = B_list
        self.Q_list = Q_list
        self.R_list = R_list
        self.x_init = x_init
        
        self.N = len(A_list)
        self.n = A_list[0].shape[0]
        self.m = B_list[0].shape[1]
    
    def build_hessian(self) -> tuple:
        """
        build condensed Hessian and gradient
        
        returns:
            H: Hessian matrix (Nm x Nm) - DENSE
            g: gradient vector (Nm)
        """
        N, n, m = self.N, self.n, self.m
        
        A_pow = [np.eye(n)]
        for _ in range(N):
            A_pow.append(A_pow[-1] @ self.A_list[0])
        
        H = np.zeros((N*m, N*m))
        g = np.zeros(N*m)
        
        for i in range(N):
            for j in range(N):
                for k in range(max(i, j) + 1, N + 1):
                    Psi_ki = A_pow[k-1-i] @ self.B_list[i]
                    Psi_kj = A_pow[k-1-j] @ self.B_list[j]
                    Q_k = self.Q_list[k]
                    
                    H[i*m:(i+1)*m, j*m:(j+1)*m] += 2 * Psi_ki.T @ Q_k @ Psi_kj
                
                if i == j:
                    H[i*m:(i+1)*m, i*m:(i+1)*m] += 2 * self.R_list[i]
            
            # gradient
            for k in range(i + 1, N + 1):
                Psi_ki = A_pow[k-1-i] @ self.B_list[i]
                Phi_k = A_pow[k]
                Q_k = self.Q_list[k]
                
                g[i*m:(i+1)*m] += 2 * Psi_ki.T @ Q_k @ Phi_k @ self.x_init
        
        # symmetrize
        H = 0.5 * (H + H.T)
        
        return H, g
    
    def analyze_structure(self, H: np.ndarray) -> dict:
        """analyze Hessian"""
        N, m = self.N, self.m
        
        # find bandwidth
        bandwidth = 0
        for i in range(N):
            for j in range(N):
                block = H[i*m:(i+1)*m, j*m:(j+1)*m]
                if np.linalg.norm(block) > 1e-10:
                    bandwidth = max(bandwidth, abs(i - j))
        
        # sparsity
        total_blocks = N * N
        nonzero_blocks = 0
        for i in range(N):
            for j in range(N):
                block = H[i*m:(i+1)*m, j*m:(j+1)*m]
                if np.linalg.norm(block) > 1e-10:
                    nonzero_blocks += 1
        
        return {
            'bandwidth': bandwidth,
            'is_dense': bandwidth == N - 1,
            'sparsity': 1.0 - (nonzero_blocks / total_blocks),
            'nonzero_blocks': nonzero_blocks,
            'total_blocks': total_blocks
        }