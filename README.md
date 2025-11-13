# LQR-BCR
Implementation of Hermitian Positive Definite (HPD) Block Cyclic Reduction for solving problems

## Overview

This project implements and analyzes block cyclic reduction methods for optimal control problems

1. **HPD Block Cyclic Reduction** (Neuenhofen 2018) - O(log N) parallel solver for block-tridiagonal systems
2. **LQR Solvers** - Standard Riccati recursion and DDP-based approaches
3. **Full KKT System Construction** - Block-tridiagonal formulation with Lagrange multipliers

## Project Goals

### Goal 1: Construct Block-Tridiagonal KKT System

- construct block-tridiagonal KKT system
- encodes LQR problem with dynamics constraints
- verified structure with bandwidth = 1
- system is indefinite as saddle-point, not compatible with HPD-BCR

### Goal 2: Implement Block Cyclic Reduction

- HPD cyclic reduction implemented
- tested on LQR-derived systems
- verified against Riccati solution
- O(log N) parallel depth

### Updated from 11/11

- Implement a generic (symmetric) BCR solver for block-tridiagonal systems in CUDA
- Think about how to leverage low-level sparsity in each block.
<img width="949" height="957" alt="image" src="https://github.com/user-attachments/assets/03499ea3-14ab-4c31-a068-a5cbe86fff0a" />

## Key Findings

### What Works
- **HPD Cyclic Reduction Algorithm**: Fully implemented and verified (Neuenhofen 2018)
- **Riccati Recursion**: Standard O(N) LQR solver working perfectly
- **Block-Tridiagonal KKT Construction**: Successfully builds full KKT system
- **DDP Newton System**: Block-tridiagonal HPD system for iterative trajectory optimization

#### Condensed formulation doesn't work
Eliminating states to get a QP in controls produces a dense Hessian with bandwidth N - 1, not block-tridiagonal. This is because each control `u_i` affects all future states `x_k` for `k > i`

#### Full KKT is indefinite
The complete KKT system with states, controls, and Lagrange multipliers is block-tridiagonal but forms a saddle point system with both positive and negative eigenvalues, incompatible with HPD cyclic reduction

#### DDP Newton is the sweet spot
Linearizing around an optimal trajectory with DDP/iLQR approach produces a block-tridiagonal HPD system—perfect for BCR
