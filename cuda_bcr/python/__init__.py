"""
CUDA-accelerated Block Cyclic Reduction solver for LQR problems.
"""

from .cuda_bcr import BCRSolver, benchmark_vs_cpu
from .utils import format_lqr_system, validate_solution

__all__ = ['BCRSolver', 'benchmark_vs_cpu', 'format_lqr_system', 'validate_solution']