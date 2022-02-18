#ifndef POWELLHYBRID_HPP
#define POWELLHYBRID_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "constants/constants.hpp"

namespace PowellHybridSolverSpace {
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
}

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

#include "dogleg.hpp"
#include "marwil.hpp"

using boost::math::constants::half;
using boost::math::constants::tenth;
using boost::math::constants::thousandth;
using boost::math::constants::ten_thousandth;

template <typename Scalar, Index RowsAtCompileTime, bool Singular>
struct PowellHybrid_Traits;

template <typename Scalar, Index _RowsAtCompileTime>
struct PowellHybrid_Traits<Scalar, _RowsAtCompileTime, true> {
public:
  static constexpr Index RowsAtCompileTime = _RowsAtCompileTime;
  template <typename Derived>
  PowellHybrid_Traits(const MatrixBase<Derived>& B) : m_B{B} {}
  // PowellHybrid_Traits(const Matrix<Scalar, _RowsAtCompileTime, _RowsAtCompileTime>& B) : m_B{B} {}
protected:
  const Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime> m_B;
};

template <typename Scalar, Index _RowsAtCompileTime>
struct PowellHybrid_Traits<Scalar, _RowsAtCompileTime, false> {
public:
  static constexpr Index RowsAtCompileTime = _RowsAtCompileTime;
};

template <typename Method, bool Singular>
class PowellHybrid : public PowellHybrid_Traits<typename Method::Scalar, Method::RowsAtCompileTime, Singular> {
public:
  using Traits = PowellHybrid_Traits<typename Method::Scalar, Method::RowsAtCompileTime, Singular>;
  typedef typename Method::Scalar Scalar;

  PowellHybrid(Method& method) : Traits(), diag{method.rows() * method.cols() + method.params()}, nfev{0}, njev{0}, iter{0}, fnorm{0}, useExternalScaling{false}, m_method{method}, n{method.rows() * method.cols() + method.params()}, ncsuc{0}, ncfail{0}, nslow1{0}, nslow2{0}, wa1{n}, wa2{n}, wa3{n}, wa4{n} {
    static_assert(Singular == false);
  }

  template <typename Derived>
  PowellHybrid(Method& method, const MatrixBase<Derived>& B) : Traits{B}, diag{method.rows() * method.cols() + method.params()}, nfev{0}, njev{0}, iter{0}, fnorm{0}, useExternalScaling{false}, m_method{method}, n{method.rows() * method.cols() + method.params()}, ncsuc{0}, ncfail{0}, nslow1{0}, nslow2{0}, wa1{n}, wa2{n}, wa3{n}, wa4{n} {
    static_assert(Singular == true);
  }

  struct Parameters {
    Parameters() : factor{100}, max_nfev{1000}, max_njev{4}, tol{Eigen::numext::sqrt(Eigen::NumTraits<Scalar>::epsilon())}, bc_tol{Eigen::numext::sqrt(Eigen::NumTraits<Scalar>::epsilon())}, epsfcn{0} {}
    Scalar factor;
    Index max_nfev;
    Index max_njev;
    Scalar tol;
    Scalar bc_tol;
    Scalar epsfcn;
  };

  PowellHybridSolverSpace::Status solveInit();
  PowellHybridSolverSpace::Status solveOneStep();
  PowellHybridSolverSpace::Status solve();

  void resetParameters(void) {parameters = Parameters();}

  void resize(Index size) {
    n = size;
    diag.resize(n);
    wa1.resize(n);
    wa2.resize(n);
    wa3.resize(n);
    wa4.resize(n);
  }

  Parameters parameters;
  Matrix<Scalar, Method::ResiduesAtCompileTime, 1> diag;
  Index nfev;
  Index njev;
  Index iter;
  Scalar fnorm;
  bool useExternalScaling;
private:
  Method& m_method;
  SparseLU<SparseMatrix<Scalar>, COLAMDOrdering<int>> m_solver;
  Index n;
  Scalar temp;
  Scalar delta;
  bool jeval;
  Index ncsuc;
  Scalar ratio;
  Scalar pnorm;
  Scalar xnorm;
  Scalar fnorm1;
  Index ncfail;
  Index nslow1;
  Index nslow2;
  Scalar actred;
  Scalar prered;
  Matrix<Scalar, Method::ResiduesAtCompileTime, 1> wa1;
  Matrix<Scalar, Method::ResiduesAtCompileTime, 1> wa2;
  Matrix<Scalar, Method::ResiduesAtCompileTime, 1> wa3;
  Matrix<Scalar, Method::ResiduesAtCompileTime, 1> wa4;

  PowellHybrid& operator=(const PowellHybrid&);
};

template <typename Method>
PowellHybrid(Method& method) -> PowellHybrid<Method, false>;

template <typename Method, typename Derived>
PowellHybrid(Method& method, const MatrixBase<Derived>& b) -> PowellHybrid<Method, true>;

template <typename Method, bool Singular>
PowellHybridSolverSpace::Status PowellHybrid<Method, Singular>::solveInit() {
  // Eigen::eigen_assert( (!useExternalScaling || diag.size() == n) && "When useExternalScaling is set, the caller must provide a valid 'diag'");
  // assert((!useExternalScaling || diag.size() == n));
  nfev = 0;
  njev = 0;

  // check the input parameters for errors.
  if (n <= 0 || parameters.tol < 0 || parameters.bc_tol < 0 || parameters.max_nfev <= 0 || parameters.max_njev <= 0 || parameters.factor <= 0) {
    return PowellHybridSolverSpace::ImproperInputParameters;
  }
  if (useExternalScaling) {
    if ((diag.array() <= 0).any()) {
      return PowellHybridSolverSpace::ImproperInputParameters;
    }
  }

  fnorm = m_method.residues.stableNorm();
  nfev = 1;
  iter = 1;
  ncsuc = 0;
  ncfail = 0;
  nslow1 = 0;
  nslow2 = 0;

  m_method.construct_global_jacobian();
  njev = 1;
  m_solver.analyzePattern(m_method.jacobian);
  m_solver.factorize(m_method.jacobian);
  if (m_solver.info() != Success) {
    return PowellHybridSolverSpace::SingularJacobian;
  }

  return PowellHybridSolverSpace::Running;
}

template <typename Method, bool Singular>
PowellHybridSolverSpace::Status PowellHybrid<Method, Singular>::solveOneStep() {
  using std::abs;

  jeval = true;

  /* Get the column wise norm of the Jacobian */
  // std::cout << "PowellHybrid: Jacobian rows x cols = " << m_method.jacobian.rows() << " x " << m_method.jacobian.cols() << std::endl;
  wa2.setZero();
  for (int k = 0; k < m_method.jacobian.outerSize(); ++k) {
    for (typename SparseMatrix<Scalar>::InnerIterator it(m_method.jacobian, k); it; ++it) {
      // std::cout << it.col() << std::endl;
      wa2(it.col()) += Eigen::numext::abs2(it.value());
    }
  }
  wa2 = wa2.cwiseSqrt();

  /* On the first iteration, and if external scaling is not used, scale
  according to the norms of the columns of the initial Jacobian */
  if (iter == 1) {
    if (!useExternalScaling) {
      diag.noalias() = wa2.unaryExpr([](Scalar w) {return (w == 0) ? 1 : w;});
      // for (Index idx = 0; idx < n; ++idx) {
      //   diag(idx) = (wa2(idx) == 0) ? 1 : wa2(idx);
      // }

      /* On the first iteration, calculate the norm of the scaled solution
      and initialize the step bound delta. */
      xnorm = diag.cwiseProduct(m_method.residues).stableNorm();
      delta = parameters.factor * xnorm;
      if (delta == 0) {
        delta = parameters.factor;
      }
    }
  }

  /* Rescale if necessary */
  if (!useExternalScaling) {
    diag = diag.cwiseMax(wa2);
  }

  while (true) {
    /* Determine the step */
    wa1.noalias() = m_solver.solve(m_method.residues); // Gauss-Newton step
    dogleg(m_method.jacobian, m_method.residues, wa1, diag, delta);

    /* store the step p and Y + p. Calculate the norm of p. */
    wa1 = -wa1;
    wa2.noalias() = m_method.Y();
    wa2 += wa1;
    /* Enforce boundary conditions at origin if a singular term is present */
    if constexpr (Singular) {
      if constexpr (Method::RowsAtCompileTime == Dynamic) {
        wa2.head(m_method.rows()) = this->m_B * wa2.head(m_method.rows());
      }
      else {
        wa2.template head<Method::RowsAtCompileTime>() = this->m_B * wa2.template head<Method::RowsAtCompileTime>();
      }
    }
    pnorm = diag.cwiseProduct(wa1).stableNorm();

    /* On the first iteration, adjust the initial step bound */
    if (iter == 1) {
      delta = std::min(delta, pnorm);
    }

    /* Evaluate the new residues and calculate the new norm. */
    wa4.noalias() = m_method(wa2);
    ++nfev;
    fnorm1 = wa4.stableNorm();

    /* Compute the scaled actual reduction */
    actred = -1;
    if (fnorm1 < fnorm) {
      actred = 1 - Eigen::numext::abs2(fnorm1 / fnorm); // Computing the 2nd pwer
    }

    /* Compute the scaled predicted reduction. */
    wa3.noalias() = m_method.jacobian * wa1;
    wa3 += m_method.residues;
    temp = wa3.stableNorm();
    prered = 0;
    if (temp < fnorm) { // Computing the 2nd power
      prered = 1 - Eigen::numext::abs2(temp / fnorm);
    }

    /* Compute the ratio of the actual to predicted reduction. */
    ratio = 0;
    if (prered > 0) {
      ratio = actred / prered;
    }

    /* Update the step bound. */
    if (ratio < tenth<Scalar>()) {
      ncsuc = 0;
      ++ncfail;
      delta = half<Scalar>() * delta;
    }
    else {
      ncfail = 0;
      ++ncsuc;
      if (ratio >= half<Scalar>() || ncsuc > 1) {
        delta = std::max(delta, 2 * pnorm);
      }
      if (abs(ratio - 1) <= tenth<Scalar>()) {
        delta = 2 * pnorm;
      }
    }

    if (ratio >= ten_thousandth<Scalar>()) {
      m_method.step(wa1);
      if constexpr (Singular) {
        m_method.apply_singular(this->m_B);
      }
      m_method.calculate();
      wa2.noalias() = diag.cwiseProduct(m_method.residues);
      // fvec = wa4;
      xnorm = wa2.stableNorm();
      fnorm = fnorm1;
      wa2.noalias() = wa4 - wa3;
      marwil(m_method.jacobian, wa1, wa2, diag);
      m_solver.factorize(m_method.jacobian);
      ++iter;
      if ((m_method.normalized_residues() < parameters.tol).all() && (m_method.bc_residues.abs() < parameters.bc_tol).all()) {
        return PowellHybridSolverSpace::Success;
      }
    }

    ++nslow1;
    if (actred >= thousandth<Scalar>()) {
      nslow1 = 0;
    }
    if (jeval) {
      ++nslow2;
    }
    if (actred >= tenth<Scalar>()) {
      nslow2 = 0;
    }

    if (nfev >= parameters.max_nfev) {
      return PowellHybridSolverSpace::TooManyFunctionEvaluations;
    }
    if (tenth<Scalar>() * std::max(tenth<Scalar>() * delta, pnorm) <= Eigen::NumTraits<Scalar>::epsilon() * xnorm) {
      return PowellHybridSolverSpace::TolTooSmall;
    }
    if (nslow2 == 5) {
      return PowellHybridSolverSpace::NotMakingProgressJacobian;
    }
    if (nslow1 == 10) {
      return PowellHybridSolverSpace::NotMakingProgressIterations;
    }

    if (ncfail == 2) {
      break;
    }

    // wa2.noalias() = wa4 - wa3;
    // marwil(m_method.jacobian, wa1, wa2, diag);
    // m_solver.factorize(m_method.jacobian);

    jeval = false;
  }
  if (njev == parameters.max_njev) {
    return PowellHybridSolverSpace::TooManyJacobianEvaluations;
  }
  m_method.construct_global_jacobian();
  m_solver.factorize(m_method.jacobian);
  ++njev;
  if (m_solver.info() != Success) {
    return PowellHybridSolverSpace::SingularJacobian;
  }
  return PowellHybridSolverSpace::Running;
}

template <typename Method, bool Singular>
PowellHybridSolverSpace::Status PowellHybrid<Method, Singular>::solve() {
  PowellHybridSolverSpace::Status status = solveInit();
  while (status == PowellHybridSolverSpace::Running) {
    status = solveOneStep();
  }
  return status;
}

} // namespace collocation

#endif // HYBRIDNONLINEARSOLVER_HPP
