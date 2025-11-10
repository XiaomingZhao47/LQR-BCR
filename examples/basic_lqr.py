# basic LQR example

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lqr.solvers import RiccatiSolver


def main():
    print("="*80)
    print("Basic LQR example with double integrator")
    print("="*80)
    
    # x = [position, velocity]
    # u = [acceleration]
    dt = 0.1
    A = np.array([[1.0, dt],
                  [0.0, 1.0]])
    B = np.array([[0.5*dt**2],
                  [dt]])
    
    # cost, penalize position and control effort
    Q = np.diag([10.0, 1.0])      # state cost
    R = np.array([[1.0]])         # control cost
    Q_f = np.diag([100.0, 10.0])  # terminal cost
    
    N = 20  # time horizon
    x_init = np.array([1.0, 0.5])  # start at position=1, velocity=0.5
    
    print(f"  Horizon: N = {N}")
    print(f"  Initial state: x_0 = {x_init}")
    
    print("\\nSolving with Riccati recursion...")
    solver = RiccatiSolver(A, B, Q, R, Q_f, N)
    states, controls, cost = solver.solve(x_init)
    
    print(f"\\nResults:")
    print(f"  Optimal cost: {cost:.4f}")
    print(f"  Final state: x_N = [{states[-1][0]:.6f}, {states[-1][1]:.6f}]")
    print(f"  Max control: {max(abs(u[0]) for u in controls):.6f}")
    
    print(f"\\nTrajectory with first 5 steps:")
    print(f"  k    x[0]      x[1]      u[0]")
    print(f"  " + "-"*35)
    for k in range(min(5, N)):
        print(f"  {k:2d}  {states[k][0]:7.4f}  {states[k][1]:7.4f}  {controls[k][0]:7.4f}")
    
    print("\\n" + "="*80)


if __name__ == '__main__':
    main()