import numpy as np
from typing import Tuple, Optional


def format_lqr_system(
    Q: np.ndarray,  # [T+1, nx, nx]
    R: np.ndarray,  # [T, nu, nu]
    A: np.ndarray,  # [T, nx, nx]
    B: np.ndarray,  # [T, nx, nu]
    q: np.ndarray,  # [T+1, nx]
    r: np.ndarray,  # [T, nu]
    x0: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Format LQR problem into KKT system for BCR.
    
    Constructs the block-tridiagonal system:
    [Q0+R0  B0^T              ] [x0]   [q0+r0]
    [B0     Q1+R1  B1^T        ] [x1]   [q1+r1]
    [       ...    ...   ...   ] [..]  = [...]
    [              BT-1    QT  ] [xT]   [qT   ]
    
    where the blocks combine state and control.
    """
    T = len(A)
    nx, nu = B[0].shape
    n = nx + nu
    
    # Construct diagonal blocks (combine Q and R)
    Q_kkt = np.zeros((T + 1, n, n))
    for k in range(T):
        Q_kkt[k, :nx, :nx] = Q[k]
        Q_kkt[k, nx:, nx:] = R[k]
    Q_kkt[T, :nx, :nx] = Q[T]
    
    # Construct off-diagonal blocks (dynamics constraints)
    B_kkt = np.zeros((T, nx, n))
    for k in range(T):
        B_kkt[k, :, :nx] = -A[k]
        B_kkt[k, :, nx:] = -B[k]
    
    # Construct RHS
    q_kkt = np.zeros((T + 1, n))
    for k in range(T):
        q_kkt[k, :nx] = q[k]
        q_kkt[k, nx:] = r[k]
    q_kkt[T, :nx] = q[T]
    
    # Handle initial condition
    if x0 is not None:
        Q_kkt[0, :nx, :nx] += np.eye(nx) * 1e6  # Large penalty
        q_kkt[0, :nx] += x0 * 1e6
    
    return Q_kkt, B_kkt, q_kkt


def validate_solution(
    x: np.ndarray,
    u: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    x0: np.ndarray,
    tolerance: float = 1e-6
) -> Tuple[bool, float]:
    """
    Validate that solution satisfies dynamics constraints.
    
    Checks: x[k+1] = A[k] * x[k] + B[k] * u[k]
    
    Returns:
        (is_valid, max_violation)
    """
    T = len(A)
    max_violation = 0.0
    
    for k in range(T):
        x_next_expected = A[k] @ x[k] + B[k] @ u[k]
        violation = np.linalg.norm(x[k + 1] - x_next_expected)
        max_violation = max(max_violation, violation)
    
    is_valid = max_violation < tolerance
    return is_valid, max_violation