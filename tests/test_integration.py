# integration tests comparing different methods

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lqr.solvers import RiccatiSolver
from kkt.ddp_newton import DDPNewtonSystem
from bcr.solver import HermitianCyclicReduction


def test_ddp_newton_matches_riccati():
    """test that DDP Newton system gives correct structure"""
    dt = 0.1
    A = np.array([[1.0, dt], [0.0, 1.0]])
    B = np.array([[0.5*dt**2], [dt]])
    Q = np.diag([10.0, 1.0])
    R = np.array([[1.0]])
    Q_f = np.diag([100.0, 10.0])
    
    N = 16
    x_init = np.array([1.0, 0.5])
    
    # solve with Riccati
    riccati = RiccatiSolver(A, B, Q, R, Q_f, N)
    states_ric, controls_ric, cost_ric = riccati.solve(x_init)
    
    # DDP Newton system
    P = riccati.get_cost_to_go()
    ddp = DDPNewtonSystem(A, B, Q, R, Q_f, N)
    newton_system = ddp.build_newton_system(states_ric, controls_ric, P)
    
    # solve with BCR
    bcr = HermitianCyclicReduction()
    delta_u, stats = bcr.solve(newton_system)
    
    max_delta = max(np.linalg.norm(du) for du in delta_u)
    
    assert max_delta < 1e-6, f"Expected zero perturbations, got {max_delta}"
    
    print(f"DDP-BCR integration test passed (max perturbation: {max_delta:.2e})")


if __name__ == '__main__':
    test_ddp_newton_matches_riccati()
    print("\\nAll integration tests passed!")

files_to_write = {
    'bcr/__init__.py': bcr_init,
    'lqr/__init__.py': lqr_init,
    'lqr/solvers.py': lqr_solvers,
    'kkt/__init__.py': kkt_init,
    'kkt/full_system.py': kkt_full_system,
    'kkt/condensed_hessian.py': kkt_condensed,
    'kkt/ddp_newton.py': kkt_ddp_newton,
    'tests/__init__.py': tests_init,
    'tests/test_riccati.py': test_riccati,
    'tests/test_bcr.py': test_bcr,
    'tests/test_kkt.py': test_kkt,
    'tests/test_integration.py': test_integration,
}

for filepath, content in files_to_write.items():
    full_path = os.path.join(base_dir, filepath)
    with open(full_path, 'w') as f:
        f.write(content)
    print(f"Created: {filepath}")

print(f"\n{'='*80}")
print("ALL MODULE FILES CREATED")
print(f"{'='*80}")
print(f"\nProject location: {base_dir}")
print("\nTo run tests:")
print(f"  cd {base_dir}")
print("  python tests/test_riccati.py")
print("  python tests/test_bcr.py")
print("  python tests/test_kkt.py")
print("  python tests/test_integration.py")