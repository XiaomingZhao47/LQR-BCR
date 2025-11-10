# tests for Riccati solver

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lqr.solvers import RiccatiSolver


def test_riccati_simple():
    """test Riccati on simple double integrator"""
    dt = 0.1
    A = np.array([[1.0, dt], [0.0, 1.0]])
    B = np.array([[0.5*dt**2], [dt]])
    Q = np.diag([10.0, 1.0])
    R = np.array([[1.0]])
    Q_f = np.diag([100.0, 10.0])
    
    N = 10
    x_init = np.array([1.0, 0.5])
    
    solver = RiccatiSolver(A, B, Q, R, Q_f, N)
    states, controls, cost = solver.solve(x_init)
    
    assert len(states) == N + 1
    assert len(controls) == N
    assert cost > 0
    
    for k in range(N):
        x_next_computed = A @ states[k] + B @ controls[k]
        assert np.allclose(x_next_computed, states[k+1], atol=1e-10)
    
    print(f"Riccati simple test passed, cost = {cost:.4f}")


def test_riccati_zero_initial():
    """test with zero initial state"""
    dt = 0.1
    A = np.array([[1.0, dt], [0.0, 1.0]])
    B = np.array([[0.5*dt**2], [dt]])
    Q = np.diag([10.0, 1.0])
    R = np.array([[1.0]])
    Q_f = np.diag([100.0, 10.0])
    
    N = 5
    x_init = np.zeros(2)
    
    solver = RiccatiSolver(A, B, Q, R, Q_f, N)
    states, controls, cost = solver.solve(x_init)
    
    assert cost < 1e-10
    for x in states:
        assert np.linalg.norm(x) < 1e-10
    for u in controls:
        assert np.linalg.norm(u) < 1e-10
    
    print("Riccati zero initial test passed")


if __name__ == '__main__':
    test_riccati_simple()
    test_riccati_zero_initial()
    print("\\nAll Riccati tests passed!")