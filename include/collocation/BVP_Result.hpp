#ifndef BVP_RESULT_HPP
#define BVP_RESULT_HPP

#include <Eigen/Dense>

using Eigen::Array;
using Eigen::ArrayBase;
using Eigen::Dynamic;
using Eigen::Index;

namespace collocation {

template <typename _Scalar, Index _ParamsAtCompileTime>
struct BVP_Result_Traits {
  typedef _Scalar Scalar;
  static constexpr Index ParamsAtCompileTime = _ParamsAtCompileTime;

  template <typename Method>
  BVP_Result_Traits(const Method& method) : p{method.p} {}

  const Array<Scalar, ParamsAtCompileTime, 1> p;
};

template <typename _Scalar>
struct BVP_Result_Traits<_Scalar, 0> {
  typedef _Scalar Scalar;
  static constexpr Index ParamsAtCompileTime = 0;

  template <typename Method>
  BVP_Result_Traits(const Method& method) {}
};

// Container to store the found solution
template <typename Method>
class BVP_Result : public BVP_Result_Traits<typename Method::Scalar, Method::ParamsAtCompileTime> {
  public:
    using Traits = BVP_Result_Traits<typename Method::Scalar, Method::ParamsAtCompileTime>;
    using Scalar = typename Traits::Scalar;
    static constexpr Index RowsAtCompileTime = Method::RowsAtCompileTime;
    static constexpr Index ColsAtCompileTime = Method::ColsAtCompileTime;
    static constexpr Index IntervalsAtCompileTime = Method::IntervalsAtCompileTime;
    using Traits::ParamsAtCompileTime;

    BVP_Result(const Method& method, const int initer, const int istatus, const std::string& imessage, const bool isuccess) : Traits{method}, sol{method}, x{method.x}, y{method.y}, yp{method.f}, residuals{method.residuals}, niter{initer}, status{istatus}, message{imessage}, success{isuccess} {}

    const typename Method::InterpolatorType sol;
    const Array<Scalar, ColsAtCompileTime, 1> x;
    const Array<Scalar, RowsAtCompileTime, ColsAtCompileTime> y;
    const Array<Scalar, RowsAtCompileTime, ColsAtCompileTime> yp;
    const Array<Scalar, IntervalsAtCompileTime, 1> residuals;
    const int niter;
    const int status;
    const std::string message;
    const bool success;
};

template <typename Method>
BVP_Result(const Method& method, const int initer, const int istatus, const std::string& imessage, const bool isuccess) -> BVP_Result<Method>;

} // namespace collocation

#endif
