# tests for KKT system construction

import numpy as np
import scipy.linalg as la
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kkt.full_system import FullKKTSystem
from kkt.condensed_hessian import CondensedHessianBuilder


def test_full_kkt_structure():
    """test full KKT has block-tridiagonal structure"""
    dt = 0.1
    A = np.array([[1.0, dt], [0.0, 1.0]])
    B = np.array([[0.5*dt**2], [dt]])
    Q = np.diag([10.0, 1.0])
    R = np.array([[1.0]])
    Q_f = np.diag([100.0, 10.0])
    
    N = 4
    x_init = np.array([1.0, 0.5])
    
    A_list = [A.copy() for _ in range(N)]
    B_list = [B.copy() for _ in range(N)]
    Q_list = [Q.copy() for _ in range(N)] + [Q_f.copy()]
    R_list = [R.copy() for _ in range(N)]
    
    kkt = FullKKTSystem(A_list, B_list, Q_list, R_list, x_init)
    system = kkt.build_full_kkt()
    
    bandwidth = system.compute_bandwidth()
    assert bandwidth <= 1, f"Expected block-tridiagonal, got bandwidth {bandwidth}"
    
    print(f"Full KKT structure test passed (bandwidth={bandwidth})")


def test_condensed_is_dense():
    """test that condensed Hessian is dense"""
    dt = 0.1
    A = np.array([[1.0, dt], [0.0, 1.0]])
    B = np.array([[0.5*dt**2], [dt]])
    Q = np.diag([10.0, 1.0])
    R = np.array([[1.0]])
    Q_f = np.diag([100.0, 10.0])
    
    N = 8
    x_init = np.array([1.0, 0.5])
    
    A_list = [A.copy() for _ in range(N)]
    B_list = [B.copy() for _ in range(N)]
    Q_list = [Q.copy() for _ in range(N)] + [Q_f.copy()]
    R_list = [R.copy() for _ in range(N)]
    
    builder = CondensedHessianBuilder(A_list, B_list, Q_list, R_list, x_init)
    H, g = builder.build_hessian()
    
    analysis = builder.analyze_structure(H)
    
    assert analysis['is_dense'], "expected dense Hessian"
    assert analysis['bandwidth'] == N - 1, f"expected bandwidth {N-1}, got {analysis['bandwidth']}"
    
    print(f"condensed Hessian density test passed (bandwidth={analysis['bandwidth']})")


if __name__ == '__main__':
    test_full_kkt_structure()
    test_condensed_is_dense()
    print("\\nAll KKT tests passed")