#ifndef NONLINEAROPTIMIZATION_DOGLEG_HPP
#define NONLINEAROPTIMIZATION_DOGLEG_HPP

namespace nonlinearbvp {

/* Calculates the dogleg step in Powell's Hybrid method */
template <typename Derived1, typename Derived2, typename Derived3, typename Derived4>
void dogleg(const SparseMatrixBase<Derived1>& jac, const MatrixBase<Derived2>& f, const MatrixBase<Derived3>& GNstep, const MatrixBase<Derived4>& diag, typename Derived3::Scalar delta) {
  using Scalar = typename Derived3::Scalar;
  Scalar GNnorm = diag.cwiseProduct(GNstep).stableNorm();
  if (GNnorm <= delta) {
    return;
  }

  // Gauss-Newton step is not acceptable
  // Calculate the steepest decent direction
  Scalar fnorm;
  Scalar temp;
  // Matrix<Scalar, Derived2::RowsAtCompileTime, 1>  SDstep = diag.cwiseInverse().cwiseProduct(jac.transpose() * f);
  Matrix<Scalar, Derived2::RowsAtCompileTime, 1> SDstep = jac.transpose() * f;
  SDstep = SDstep.cwiseQuotient(diag);

  // Calculate the norm of the scaled steepest decent step and test for the
  // special case in which the scaled steepest decent step is zero.
  Scalar SDnorm = SDstep.stableNorm();
  Scalar scaledSDnorm = 0;
  Scalar alpha = delta / GNnorm;
  if (SDnorm == 0) {
    goto algo_end;
  }

  // SDstep.array() /= (diag * SDnorm).array();
  SDstep = SDstep.cwiseQuotient(SDnorm * diag);
  temp = (jac * SDstep).stableNorm();
  scaledSDnorm = SDnorm / (temp * temp);

  alpha = 0;
  if (scaledSDnorm >= delta) {
    goto algo_end;
  }

  // The steepest decent step is not acceptable.
  // Finally, calculate the point along the dogleg
  // at which the quadratic is minimized.
  fnorm = f.stableNorm();
  temp = fnorm / SDnorm * (fnorm / GNnorm) * (scaledSDnorm / delta);
  temp = temp - delta / GNnorm * Eigen::numext::abs2(scaledSDnorm / delta) + sqrt(Eigen::numext::abs2(temp - delta / GNnorm) + (1 - Eigen::numext::abs2(delta / GNnorm)) * (1 - Eigen::numext::abs2(scaledSDnorm / delta)));
  alpha = delta / GNnorm * (1 - Eigen::numext::abs2(scaledSDnorm / delta)) / temp;

algo_end:
  temp = (1 - alpha) * std::min(scaledSDnorm, delta);
  const_cast<MatrixBase<Derived3>&>(GNstep) = temp * SDstep + alpha * GNstep;
  return;
}

} // namespace nonlinearbvp

#endif // NONLINEAROPTIMIZATION_DOGLEG_HPP
