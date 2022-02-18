#ifndef COLLOCATION_PRINT_FUNCTIONS_HPP
#define COLLOCATION_PRINT_FUNCTIONS_HPP
#include <iostream>
#include <iomanip>
#include <limits>

#ifndef BVPSOLVERSPACE
#define BVPSOLVERSPACE
#include <string>

namespace BVPSolverSpace {
  enum Status {Success = 1,
               MaximumNodesExceeded = 2,
               SingularJacobian = 3,
               TolTooSmall = 4,
               NotMakingProgressResiduals = 5
               };
}
#endif

namespace collocation { namespace detail {

// Termination messages for collocation algorithm
// std::string TERMINATION_MESSAGES[4] = {"The algorithm converged to the desired accuracy.",
//                                        "The maximum number of mesh nodes is exceeded.",
//                                        "A singular Jacobian encoutered when solving the collocation system.",
//                                        "The solver was unable to satisfy boundary conditions tolerance on iteration 10."};

std::string Termination_Messages(const BVPSolverSpace::Status status) {
  std::string message;
  switch (status) {
    case BVPSolverSpace::Success : {
      message = "The algorithm converged to the desired accuracy";
      break;
    }
    case BVPSolverSpace::MaximumNodesExceeded: {
      message = "The maximum number of mesh nodes was exceeded.";
      break;
    }
    case BVPSolverSpace::SingularJacobian: {
      message = "A singular Jacobian was encountered when solving the collocation system.";
      break;
    }
    case BVPSolverSpace::TolTooSmall: {
      message = "The solver was unable to satisfy boundary conditions on final iteration.";
      break;
    }
    case BVPSolverSpace::NotMakingProgressResiduals: {
      message = "Maximum residual did not decrease after two iterations.";
      break;
    }
    default: {
      message = "I fucked up...";
    }
  }
  return message;
}

// Prints the evaluation header
void print_iteration_header() {
  std::cout << std::setprecision(4) << std::scientific;
  std::cout << std::string(90, '-') << std::endl;
  std::cout << "| Iteration | Max residual | Max BC residual | Total nodes | Nodes added | Nodes Removed |" << std::endl;
  std::cout << std::string(90, '-') << std::endl;
};

// Prints current progress at each iteration
template <typename Scalar>
void print_iteration_progress(const int iteration, const Scalar residual, const Scalar bc_residual, const int total_nodes, const int nodes_added, const int nodes_removed) {
  std::cout << "| " << std::setw(9) << iteration << " ";
  std::cout << "| " << std::setw(12) << residual << " ";
  std::cout << "| " << std::setw(15) << bc_residual << " ";
  std::cout << "| " << std::setw(11) << total_nodes << " ";
  std::cout << "| " << std::setw(11) << nodes_added << " ";
  std::cout << "| " << std::setw(13) << nodes_removed << " |" << std::endl;
  std::cout << std::string(90, '-') << std::endl;
};

template <typename Scalar>
void print_result(const BVPSolverSpace::Status status, const int iteration, const Scalar max_res, const Scalar max_bc_res, int nodes) {
  std::cout << std::setprecision(std::numeric_limits<Scalar>::max_digits10) << std::scientific;
  switch (status) {
    case BVPSolverSpace::Success:
    {
      std::cout << "Solved in " << std::to_string(iteration) << " iterations, number of nodes " << std::to_string(nodes) << "." << std::endl;
      std::cout << "Maximum relative residual: " << max_res << std::endl;
      std::cout << "Maximum boundary residual: " << max_bc_res << std::endl;
      break;
    }
    case BVPSolverSpace::SingularJacobian: {
      std::cout << "Singular Jacobian encountered when solving collocation system on iteration " << std::to_string(iteration) << "." << std::endl;
      std::cout << "Maximum relative residual: " << max_res << std::endl;
      std::cout << "Maximum boundary residual: " << max_bc_res << std::endl;
      break;
    }
    case BVPSolverSpace::MaximumNodesExceeded: {
      std::cout << "Number of nodes is exceeded after iteration " << std::to_string(iteration) << "." << std::endl;
      std::cout << "Maximum relative residual: " << max_res << std::endl;
      std::cout << "Maximum boundary residual: " << max_bc_res << std::endl;
      break;
    }
    case BVPSolverSpace::TolTooSmall: {
      std::cout << "The solver was unable to satisfy boundary conditions tolerance on iteration " << std::to_string(iteration) << "." << std::endl;
      std::cout << "Maximum relative residual: " << max_res << std::endl;
      std::cout << "Maximum boundary residual: " << max_bc_res << std::endl;
      break;
    }
    case BVPSolverSpace::NotMakingProgressResiduals: {
      std::cout << "Max residual did not decrease on iteration " << std::to_string(iteration) << "." << std::endl;
      std::cout << "Maximum relative residual: " << max_res << std::endl;
      std::cout << "Maximum boundary residual: " << max_bc_res << std::endl;
      break;
    }
    default: std::cout << "I fucked up..." << std::endl;
  }
  std::cout << std::resetiosflags( std::cout.flags() ) << std::setprecision(6);
}

} // namespace detail
} // namespace collocation

#endif // COLLOCATION_PRINT_FUNCTIONS_HPP
