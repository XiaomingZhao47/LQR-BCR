"""
CUDA-accelerated Block Cyclic Reduction solver for LQR problems.
"""

try:
    from cuda_bcr._cuda_bcr import BCRSolver as _BCRSolver
    _cuda_available = True
except ImportError as e:
    _BCRSolver = None
    _cuda_available = False
    import warnings
    warnings.warn(f"CUDA BCR module not available: {e}")

# Re-export from the Python wrapper
from .solver import BCRSolver
from .utils import format_lqr_system, validate_solution, benchmark_vs_cpu

__version__ = "0.1.0"
__all__ = ['BCRSolver', 'format_lqr_system', 'validate_solution', 'benchmark_vs_cpu']

def is_cuda_available():
    """Check if CUDA extension is available."""
    return _cuda_available
