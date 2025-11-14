#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "bcr_solver.cuh"

namespace py = pybind11;

class PyBCRSolver {
public:
    PyBCRSolver(int horizon, int state_dim, int control_dim,
                int max_iter = 100, double tol = 1e-6, bool verbose = false) {
        BCRConfig config;
        config.horizon = horizon;
        config.state_dim = state_dim;
        config.control_dim = control_dim;
        config.max_iter = max_iter;
        config.tolerance = tol;
        config.use_riccati = false;
        config.verbose = verbose;
        
        solver_ = new CUDABCRSolver(config);
    }
    
    ~PyBCRSolver() {
        delete solver_;
    }
    
    py::array_t<double> solve(
        py::array_t<double> Q,
        py::array_t<double> B,
        py::array_t<double> q
    ) {
        auto Q_buf = Q.request();
        auto B_buf = B.request();
        auto q_buf = q.request();
        
        int T = Q_buf.shape[0] - 1;
        int n = Q_buf.shape[1];
        
        // Allocate output
        py::array_t<double> x({T + 1, n});
        auto x_buf = x.request();
        
        // Solve
        BCRStatus status = solver_->solveHost(
            static_cast<double*>(Q_buf.ptr),
            static_cast<double*>(B_buf.ptr),
            static_cast<double*>(q_buf.ptr),
            static_cast<double*>(x_buf.ptr)
        );
        
        if (status != BCR_SUCCESS) {
            throw std::runtime_error("BCR solve failed");
        }
        
        return x;
    }
    
    py::dict get_stats() {
        BCRStats stats = solver_->getStats();
        py::dict d;
        d["num_stages"] = stats.num_stages;
        d["forward_time_ms"] = stats.forward_time_ms;
        d["backward_time_ms"] = stats.backward_time_ms;
        d["total_time_ms"] = stats.total_time_ms;
        return d;
    }

private:
    CUDABCRSolver* solver_;
};

PYBIND11_MODULE(_cuda_bcr, m) {
    m.doc() = "CUDA Block Cyclic Reduction solver for LQR";
    
    py::class_<PyBCRSolver>(m, "BCRSolver")
        .def(py::init<int, int, int, int, double, bool>(),
             py::arg("horizon"),
             py::arg("state_dim"),
             py::arg("control_dim"),
             py::arg("max_iter") = 100,
             py::arg("tolerance") = 1e-6,
             py::arg("verbose") = false)
        .def("solve", &PyBCRSolver::solve,
             "Solve LQR problem using BCR",
             py::arg("Q"), py::arg("B"), py::arg("q"))
        .def("get_stats", &PyBCRSolver::get_stats,
             "Get solver statistics");
}