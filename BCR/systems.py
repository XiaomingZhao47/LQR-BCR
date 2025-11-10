import os

# block-tridiagonal system data structures for cyclic reduction

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import scipy.linalg as la


@dataclass
class BlockTridiagonalSystem:
    """
    block-tridiagonal system: Ax = b
    
    structure
        [A_0  B_0^T      0    ...    0  ]
        [B_0  A_1   B_1^T    ...    0  ]
        [ 0   B_1    A_2     ...    0  ]
        [            ...            B_{N - 2}^T]
        [ 0   ...    0   B_{N - 2}  A_{N - 1}  ]
    
    attributes
        A_blocks: diagonal blocks [A_0, A_1, ..., A_{N - 1}]
        B_blocks: fff-diagonal blocks [B_0, B_1, ..., B_{N - 2}]
        b_blocks: RHS vectors [b_0, b_1, ..., b_{N - 1}]
    """
    A_blocks: List[np.ndarray]  # diagonal blocks
    B_blocks: List[np.ndarray]  # off-diagonal 
    b_blocks: List[np.ndarray]  # RHS vectors
    
    @property
    def N(self) -> int:
        """num of blocks"""
        return len(self.A_blocks)
    
    @property
    def block_size(self) -> int:
        """size of each block"""
        return self.A_blocks[0].shape[0]
    
    def verify_hermitian(self) -> bool:
        """check if diagonal blocks are Hermitian"""
        for A in self.A_blocks:
            if not np.allclose(A, A.conj().T, atol=1e-10):
                return False
        return True
    
    def verify_positive_definite(self) -> Tuple[bool, float, float]:
        """check if system is positive"""
        A_dense, _ = self.to_dense()
        eigs = la.eigvalsh(A_dense)
        return eigs.min() > 0, eigs.min(), eigs.max()
    
    def to_dense(self) -> Tuple[np.ndarray, np.ndarray]:
        """convert to dense matrix"""
        N = self.N
        bs = self.block_size
        n = N * bs
        
        A = np.zeros((n, n))
        b = np.zeros(n)
        
        for i in range(N):
            idx = slice(i * bs, (i + 1) * bs)
            A[idx, idx] = self.A_blocks[i]
            b[idx] = self.b_blocks[i]
            
            if i < N - 1:
                idx_next = slice((i + 1) * bs, (i + 2) * bs)
                # symmetric B_i connects blocks i and i + 1
                A[idx, idx_next] = self.B_blocks[i].conj().T
                A[idx_next, idx] = self.B_blocks[i]
        
        return A, b
    
    def compute_bandwidth(self) -> int:
        """compute block bandwidth"""
        A_dense, _ = self.to_dense()
        bs = self.block_size
        
        bandwidth = 0
        for i in range(self.N):
            for j in range(self.N):
                block = A_dense[i*bs:(i+1)*bs, j*bs:(j+1)*bs]
                if np.linalg.norm(block) > 1e-10:
                    bandwidth = max(bandwidth, abs(i - j))
        
        return bandwidth
