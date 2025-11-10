# LQR solvers, Riccati recursion and DDP

import numpy as np
import scipy.linalg as la
from typing import Tuple, List


class RiccatiSolver:
    """
    standard Riccati recursion for LQR
    
    solves min sum_k [x_k^T Q_k x_k + u_k^T R_k u_k] + x_N^T Q_f x_N
            s.t. x_{k+1} = A_k x_k + B_k u_k, x_0 = x_init
    
    complexity: O(N) in time, sequential
    """
    
    def __init__(self, A, B, Q, R, Q_f, N):
        """
        args:
            A: State transition matrix (n x n)
            B: Control input matrix (n x m)
            Q: State cost matrix (n x n)
            R: Control cost matrix (m x m)
            Q_f: Terminal cost matrix (n x n)
            N: Time horizon
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.Q_f = Q_f
        self.N = N
        
        self.n = A.shape[0]  # state dimension
        self.m = B.shape[1]  # control dimension
        
        self.P = None
        self.K = None
    
    def solve(self, x_init: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
        """
        solve LQR problem
        
        args:
            x_init: Initial state (n,)
            
        returns:
            states: list of states [x_0, x_1, ..., x_N]
            controls: list of controls [u_0, u_1, ..., u_{N-1}]
            cost
        """
        # cost-to-go and gains
        self._backward_pass()
        
        # optimal trajectory
        states, controls = self._forward_pass(x_init)
        
        # cost
        cost = self._compute_cost(states, controls)
        
        return states, controls, cost
    
    def _backward_pass(self):
        """backward Riccati recursion"""
        N = self.N
        
        self.P = [None] * (N + 1)
        self.K = [None] * N
        
        self.P[N] = self.Q_f
        
        # backward recursion
        for k in range(N - 1, -1, -1):
            # compute optimal gain
            temp = self.R + self.B.T @ self.P[k+1] @ self.B
            self.K[k] = -la.solve(temp, self.B.T @ self.P[k+1] @ self.A)
            
            A_cl = self.A + self.B @ self.K[k]
            self.P[k] = (self.Q + self.K[k].T @ self.R @ self.K[k] + 
                        A_cl.T @ self.P[k+1] @ A_cl)
    
    def _forward_pass(self, x_init: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """forward simulation with optimal controls"""
        states = [x_init]
        controls = []
        
        for k in range(self.N):
            u_k = self.K[k] @ states[k]
            controls.append(u_k)
            
            x_next = self.A @ states[k] + self.B @ u_k
            states.append(x_next)
        
        return states, controls
    
    def _compute_cost(self, states: List[np.ndarray], controls: List[np.ndarray]) -> float:
        """compute total cost"""
        cost = 0.0
        
        for k in range(self.N):
            cost += states[k].T @ self.Q @ states[k]
            cost += controls[k].T @ self.R @ controls[k]
        
        cost += states[self.N].T @ self.Q_f @ states[self.N]
        
        return float(cost)
    
    def get_feedback_gains(self) -> List[np.ndarray]:
        """return computed feedback gains"""
        if self.K is None:
            raise RuntimeError("call solve() first")
        return self.K
    
    def get_cost_to_go(self) -> List[np.ndarray]:
        """return cost-to-go matrices"""
        if self.P is None:
            raise RuntimeError("call solve() first")
        return self.P


class DDPSolver:
    """
    DDP solver
    
    refines trajectory using second-order approximation
    """
    
    def __init__(self, A, B, Q, R, Q_f, N, max_iters=10):
        """
        args:
            A:         state transition matrix
            B:         control input matrix
            Q:         state cost matrix
            R:         control cost matrix
            Q_f:       terminal cost matrix
            N:         time horizon
            max_iters: maximum DDP iterations
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.Q_f = Q_f
        self.N = N
        self.max_iters = max_iters
        
        self.n = A.shape[0]
        self.m = B.shape[1]
    
    def solve(self, x_init: np.ndarray, tol: float = 1e-6) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
        """
        solve via DDP iterations
        
        args:
            x_init: Initial state
            tol: Convergence tolerance
            
        returns:
            states: Optimal state trajectory
            controls: Optimal control trajectory
            cost: Optimal cost
        """
        riccati = RiccatiSolver(self.A, self.B, self.Q, self.R, self.Q_f, self.N)
        states, controls, cost = riccati.solve(x_init)
        
        print(f"Initial cost (Riccati): {cost:.6f}")
        
        # DDP iterations
        for iter in range(self.max_iters):
            P = self._backward_pass(states, controls)
            
            states_new, controls_new, cost_new = self._forward_pass(x_init, states, controls, P)
            
            cost_improvement = cost - cost_new
            print(f"Iteration {iter+1}: cost = {cost_new:.6f}, improvement = {cost_improvement:.2e}")
            
            if abs(cost_improvement) < tol:
                print("converged")
                break
            
            states, controls, cost = states_new, controls_new, cost_new
        
        return states, controls, cost
    
    def _backward_pass(self, states, controls):
        """backward pass, compute local value function"""
        N = self.N
        P = [None] * (N + 1)
        P[N] = self.Q_f
        
        for k in range(N - 1, -1, -1):
            # Linearized dynamics; for LQR, already linear
            # V_xx = Q + A^T P_{k+1} A + ...
            # for LQR this reduces to Riccati
            temp = self.R + self.B.T @ P[k+1] @ self.B
            K_k = -la.solve(temp, self.B.T @ P[k+1] @ self.A)
            A_cl = self.A + self.B @ K_k
            P[k] = self.Q + K_k.T @ self.R @ K_k + A_cl.T @ P[k+1] @ A_cl
        
        return P
    
    def _forward_pass(self, x_init, states_old, controls_old, P):
        """forward pass, apply control updates"""
        # for LQR, use the Riccati gains
        # for nonlinear systems, would do line search
        
        states_new = [x_init]
        controls_new = []
        
        for k in range(self.N):
            temp = self.R + self.B.T @ P[k+1] @ self.B
            K_k = -la.solve(temp, self.B.T @ P[k+1] @ self.A)
            
            u_k = K_k @ states_new[k]
            controls_new.append(u_k)
            
            x_next = self.A @ states_new[k] + self.B @ u_k
            states_new.append(x_next)
        
        cost = sum(states_new[k].T @ self.Q @ states_new[k] + 
                   controls_new[k].T @ self.R @ controls_new[k] 
                   for k in range(self.N))
        cost += states_new[self.N].T @ self.Q_f @ states_new[self.N]
        
        return states_new, controls_new, float(cost)