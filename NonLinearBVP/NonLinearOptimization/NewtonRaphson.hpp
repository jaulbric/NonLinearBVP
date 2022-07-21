#ifndef NEWTONRAPHSON_HPP
#define NEWTONRAPHSON_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "NonLinearBVP/NonLinearOptimization/marwil.hpp"
#include "NonLinearBVP/constants/constants.hpp"

namespace nonlinearbvp {

namespace NewtonRaphsonSolverSpace {
  enum Status {
    Running = -1,
    ImproperInputParameters = 0,
    Success = 1,
    TooManyFunctionEvaluations = 2,
    TooManyJacobianEvaluations = 3,
    TolTooSmall = 4,
    NotMakingProgressJacobian = 5,
    NotMakingProgressIterations = 6,
    SingularJacobian = 7,
    UserAsked = 8
  };
} // namespace NewtonRaphsonSolverSpace

namespace collocation {

using Eigen::Array;
using Eigen::Matrix;
using Eigen::ArrayBase;
using Eigen::MatrixBase;
using Eigen::Index;
using Eigen::SparseMatrix;
using Eigen::SparseMatrixBase;
using Eigen::SparseLU;
using Eigen::COLAMDOrdering;
using Eigen::Success;
using Eigen::Dynamic;

using boost::math::constants::half;
using boost::math::constants::fifth;

template <typename Scalar, Index RowsAtCompileTime, bool Singular>
class NewtonRaphson_Traits;

template <typename Scalar, Index _RowsAtCompileTime>
class NewtonRaphson_Traits<Scalar, _RowsAtCompileTime, true> {
public:
  static constexpr Index RowsAtCompileTime = _RowsAtCompileTime;
  template <typename Derived>
  NewtonRaphson_Traits(const MatrixBase<Derived>& B) : m_B{B} {}
protected:
  const Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime> m_B;
};

template <typename Scalar, Index _RowsAtCompileTime>
class NewtonRaphson_Traits<Scalar, _RowsAtCompileTime, false> {
public:
  static constexpr Index RowsAtCompileTime = _RowsAtCompileTime;
};

template <typename Method, bool Singular>
class NewtonRaphson : public NewtonRaphson_Traits<typename Method::Scalar, Method::RowsAtCompileTime, Singular> {
public:
  using Traits = NewtonRaphson_Traits<typename Method::Scalar, Method::RowsAtCompileTime, Singular>;
  typedef typename Method::Scalar Scalar;
  // typedef typename SparseQR<SparseMatrix<Scalar>, COLAMDOrdering<int>> SolverType;

  NewtonRaphson(Method& method) : Traits(), nfev{0}, njev{0}, iter{0}, m_method{method}, n{method.rows() * method.cols() + method.params()}, recompute_jac{true}, wa1{n}, wa2{n}, wa3{n}, wa4{n} {}

  template <typename Derived>
  NewtonRaphson(Method& method, const MatrixBase<Derived>& B) : Traits{B}, nfev{0}, njev{0}, iter{0}, m_method{method}, n{method.rows() * method.cols() + method.params()}, recompute_jac{true}, wa1{n}, wa2{n}, wa3{n}, wa4{n} {
    // std::cout << "NewtonRaphson constructor: " << &B << std::endl;
  }

  struct Parameters {
    Parameters() : max_njev{4}, max_iterations{8}, max_trials{4}, tol{Eigen::numext::sqrt(Eigen::NumTraits<Scalar>::epsilon())}, bc_tol{Eigen::numext::sqrt(Eigen::NumTraits<Scalar>::epsilon())} {}

    int max_njev;
    int max_iterations;
    int max_trials;
    Scalar tol;
    Scalar bc_tol;
  };

  NewtonRaphsonSolverSpace::Status solveInit();
  NewtonRaphsonSolverSpace::Status solveOneStep();
  NewtonRaphsonSolverSpace::Status solve();

  void resetParameters(void) {parameters = Parameters();}

  void resize(Index size) {
    n = size;
    wa1.resize(n);
    wa2.resize(n);
    wa3.resize(n);
    wa4.resize(n);
  }

  Parameters parameters;
  int nfev;
  int njev;
  int iter;

private:
  Method& m_method;
  SparseLU<SparseMatrix<Scalar>, COLAMDOrdering<int>> m_solver;
  Index n;
  static constexpr Scalar m_sigma = fifth<Scalar>();
  static constexpr Scalar m_tau = half<Scalar>();
  bool recompute_jac;
  Matrix<Scalar, Method::ResiduesAtCompileTime, 1> wa1;
  Matrix<Scalar, Method::ResiduesAtCompileTime, 1> wa2;
  Matrix<Scalar, Method::ResiduesAtCompileTime, 1> wa3;
  Matrix<Scalar, Method::ResiduesAtCompileTime, 1> wa4;
  Scalar m_cost;

  NewtonRaphson& operator=(const NewtonRaphson&);
};

template <typename Method, typename Derived>
NewtonRaphson(Method& method, const MatrixBase<Derived>& B) -> NewtonRaphson<Method, true>;

template <typename Method>
NewtonRaphson(Method& method) -> NewtonRaphson<Method, false>;

template <typename Method, bool Singular>
NewtonRaphsonSolverSpace::Status NewtonRaphson<Method, Singular>::solveInit() {
  nfev = 1;
  iter = 0;
  if (parameters.tol < 0 || parameters.bc_tol < 0 || parameters.max_njev <= 0 || parameters.max_iterations <= 0 || parameters.max_trials <= 0) {
    return NewtonRaphsonSolverSpace::ImproperInputParameters;
  }
  m_method.construct_global_jacobian();
  m_solver.analyzePattern(m_method.jacobian);
  m_solver.factorize(m_method.jacobian);
  if (m_solver.info() != Success) {
    return NewtonRaphsonSolverSpace::SingularJacobian;
  }
  // std::cout << m_method.jacobian << std::endl;
  // m_step = -m_solver.solve(m_method.residues);
  // m_cost = m_step.squaredNorm();
  // recompute_jac = false;
  njev = 1;
  return NewtonRaphsonSolverSpace::Running;
}

template <typename Method, bool Singular>
NewtonRaphsonSolverSpace::Status NewtonRaphson<Method, Singular>::solveOneStep() {
  while (iter < parameters.max_iterations) {
    wa1.noalias() = m_solver.solve(m_method.residues);
    m_cost = wa1.squaredNorm();
    wa1 = -wa1;

    int trial;
    Scalar alpha = 1;
    Scalar cost_new;
    for (trial = 0; trial < parameters.max_trials; ++trial) {
      wa2.noalias() = m_method.Y() + wa1;
      if constexpr (Singular) {
        if constexpr (Method::RowsAtCompileTime == Dynamic) {
          wa2.head(m_method.rows()) = this->m_B * wa2.head(m_method.rows());
        }
        else {
          wa2.template head<Method::RowsAtCompileTime>() = this->m_B * wa2.template head<Method::RowsAtCompileTime>();
        }
      }
      // wa4.noalias() = m_method(wa2);
      m_method(wa2, wa4);
      wa3.noalias() = m_solver.solve(wa4);
      cost_new = wa3.squaredNorm();
      if (cost_new < (1 - 2 * alpha * m_sigma) * m_cost) {
        break;
      }
      wa1 *= m_tau;
      alpha *= m_tau;
    }

    wa3 = m_method.residues + m_method.jacobian * wa1;

    m_method.step(wa1);
    if constexpr (Singular) {
      m_method.apply_singular(this->m_B);
    }
    m_method.calculate();
    ++nfev;

    // Stopping criterion
    if ((m_method.normalized_residues() < parameters.tol).all() && (m_method.bc_residues.abs() < parameters.bc_tol).all()) {
      return NewtonRaphsonSolverSpace::Success;
    }
    if (trial > 0) {
      // recompute full Jacobian;
      ++iter;
      break;
    }
    else {
      // update Jacobian using Marwil method;
      wa2 = wa4 - wa3;
      marwil(m_method.jacobian, wa1, wa2);
      m_solver.factorize(m_method.jacobian);
      ++iter;
    }
  }
  if (njev == parameters.max_njev) {
    return NewtonRaphsonSolverSpace::TooManyJacobianEvaluations;
  }
  m_method.construct_global_jacobian();
  m_solver.factorize(m_method.jacobian);
  ++njev;
  if (m_solver.info() != Success) {
    return NewtonRaphsonSolverSpace::SingularJacobian;
  }
  return NewtonRaphsonSolverSpace::Running;
}


  // if (recompute_jac) {
  //   m_method.construct_global_jacobian();
  //   m_solver.factorize(m_method.jacobian);
  //   if (m_solver.info() != Success) {
  //     return NewtonRaphsonSolverSpace::SingularJacobian;
  //   }
  //   m_step = -m_solver.solve(m_method.residues);
  //   m_cost = m_step.squaredNorm();
  //   ++njev;
  // }

//   m_method.step(m_step);
//   if constexpr (Singular) {
//     m_method.apply_singular(this->m_B);
//   }
//   m_method.calculate();
//   ++nfev;
//
//   // Backtracking linesearch
//   Scalar cost_new;
//   Scalar alpha = 1;
//   int trial;
//   for (trial = 0; trial < parameters.max_trials; ++trial) {
//     m_step = m_solver.solve(m_method.residues);
//     cost_new = m_step.squaredNorm();
//     m_step = m_step
//     if (cost_new < (1 - 2 * alpha * m_sigma) * m_cost) {
//       break;
//     }
//     m_step *= (1 - m_tau) * alpha;
//     // m_method.step((1 - m_tau) * alpha * m_step);
//     m_method.step(m_step);
//     if constexpr (Singular) {
//       m_method.apply_singular(this->m_B);
//     }
//     m_method.calculate();
//     ++nfev;
//     alpha *= m_tau;
//   } // Backtracking linesearch;
//   ++iter;
//
//
//   // Stopping criterion
//   if ((m_method.normalized_residues() < parameters.tol / 20).all() && (m_method.bc_residues < parameters.bc_tol).all()) {
//     return NewtonRaphsonSolverSpace::Success;
//   }
//   if (njev >= parameters.max_njev) {
//     return NewtonRaphsonSolverSpace::TooManyJacobianEvaluations;
//   }
//   if (iter >= parameters.max_iterations) {
//     return NewtonRaphsonSolverSpace::NotMakingProgressIterations;
//   }
//
//   if (trial == 0) {
//     m_cost = cost_new;
//     recompute_jac = false;
//   }
//   else {
//     recompute_jac = true;
//   }
//   return NewtonRaphsonSolverSpace::Running;
// }

template <typename Method, bool Singular>
NewtonRaphsonSolverSpace::Status NewtonRaphson<Method, Singular>::solve() {
  NewtonRaphsonSolverSpace::Status status = solveInit();
  while (status == NewtonRaphsonSolverSpace::Running) {
    status = solveOneStep();
  }
  return status;
}

} // namespace collocation
} // namespace nonlinearbvp

#endif // NEWTONRAPHSON_HPP
