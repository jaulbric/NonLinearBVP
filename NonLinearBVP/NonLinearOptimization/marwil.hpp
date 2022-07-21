#ifndef NONLINEAROPTIMIZATION_MARWIL_HPP
#define NONLINEAROPTIMIZATION_MARWIL_HPP
/* Update the Jacobian along the direction of the step, being careful not to
change the sparsity structure of the Jacobian.

Marwil, E. (1979). Convergence Results for Schubert’s Method for Solving Sparse
Nonlinear Equations. SIAM Journal on Numerical Analysis, 16(4), 588–604.
http://www.jstor.org/stable/2156530 */

namespace nonlinearbvp {

template <typename Scalar, typename Derived1, typename Derived2, typename Derived3>
void marwil(const SparseMatrix<Scalar>& jac, const MatrixBase<Derived1>& p, const MatrixBase<Derived2>& fpred, const MatrixBase<Derived3>& diag) {
  using VectorType = Matrix<Scalar, Derived1::SizeAtCompileTime, 1>;
  VectorType s2norm = VectorType::Zero(p.size());
  for (int j = 0; j < jac.outerSize(); ++j) {
    for (typename SparseMatrix<Scalar>::InnerIterator it(jac, j); it; ++it) {
      s2norm(it.row()) += Eigen::numext::abs2(diag(it.col()) * p(it.col()));
    }
  }
  for (int j = 0; j < jac.outerSize(); ++j) {
    for (typename SparseMatrix<Scalar>::InnerIterator it(jac, j); it; ++it) {
      if (s2norm(it.row()) == 0) {
        continue;
      }
      it.valueRef() += fpred(it.row()) * diag(it.col()) * diag(it.col()) * p(it.col()) / s2norm(it.row());
    }
  }
}

// Overload where diag is assumed to be a vector of ones
template <typename Scalar, typename Derived1, typename Derived2>
void marwil(const SparseMatrix<Scalar>& jac, const MatrixBase<Derived1>& p, const MatrixBase<Derived2>& fpred) {
  using VectorType = Matrix<Scalar, Derived1::SizeAtCompileTime, 1>;
  VectorType s2norm = VectorType::Zero(p.size());
  for (int j = 0; j < jac.outerSize(); ++j) {
    for (typename SparseMatrix<Scalar>::InnerIterator it(jac, j); it; ++it) {
      s2norm(it.row()) += Eigen::numext::abs2(p(it.col()));
    }
  }
  for (int j = 0; j < jac.outerSize(); ++j) {
    for (typename SparseMatrix<Scalar>::InnerIterator it(jac, j); it; ++it) {
      if (s2norm(it.row()) == 0) {
        continue;
      }
      it.valueRef() += fpred(it.row()) * p(it.col()) / s2norm(it.row());
    }
  }
}

} // namespace nonlinearbvp

#endif // NONLINEAROPTIMIZATION_MARWIL_HPP
