#!/usr/bin/env python3
"""
Benchmark CUDA BCR against CPU Riccati recursion.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cuda_bcr import BCRSolver, is_cuda_available
from cuda_bcr.utils import benchmark_vs_cpu

if not is_cuda_available():
    print("ERROR: CUDA BCR module not available")
    sys.exit(1)

def main():
    print("=" * 60)
    print("CUDA BCR vs CPU Riccati Benchmark")
    print("=" * 60)
    
    # Test different problem sizes
    horizons = [10, 20, 50, 100, 200, 500]
    state_dims = [4, 8, 12]
    control_dim = 2
    
    results = {}
    
    for nx in state_dims:
        print(f"\nState dimension: {nx}, Control dimension: {control_dim}")
        print("-" * 60)
        
        speedups = []
        
        for T in horizons:
            print(f"Horizon {T:4d}... ", end='', flush=True)
            
            result = benchmark_vs_cpu(T, nx, control_dim, n_trials=10)
            
            speedups.append(result['speedup'])
            
            print(f"CPU: {result['cpu_mean_ms']:6.2f} ms, "
                  f"CUDA: {result['cuda_mean_ms']:6.2f} ms, "
                  f"Speedup: {result['speedup']:5.2f}x, "
                  f"Error: {result['relative_error']:.2e}")
        
        results[nx] = speedups
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for nx, speedups in results.items():
        plt.plot(horizons, speedups, marker='o', label=f'nx={nx}')
    
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Horizon T')
    plt.ylabel('Speedup (CPU time / CUDA time)')
    plt.title('CUDA BCR vs CPU Riccati: Speedup')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('bcr_speedup.png', dpi=150)
    print("\nPlot saved to bcr_speedup.png")


if __name__ == '__main__':
    main()