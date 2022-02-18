#ifndef COLLOCATION_TOOLS_HPP
#define COLLOCATION_TOOLS_HPP

#include <algorithm>
#include <iostream>
#include <exception>
#include <Eigen/Core>
#include <Eigen/Dense>
// #include <Eigen/CXX11/Tensor>
#include "constants/constants.hpp"

using Eigen::Dynamic;
using Eigen::Index;
using Eigen::Array;
using Eigen::ArrayBase;
using Eigen::DenseBase;
using Eigen::seq;
using Eigen::last;
using Eigen::Tensor;
using Eigen::TensorRef;
using Eigen::TensorBase;
using Eigen::array;
using Eigen::IndexPair;
using Eigen::ReadOnlyAccessors;

using boost::math::constants::ln_three;

namespace collocation { namespace tools {

// quick batched matrix multiplication function for 3D tensors
template <typename Real>
inline auto matmul(const Tensor<Real, 3>& a, const Tensor<Real, 3>& b) {
  return a.contract(b, array<IndexPair<Index>, 1>({IndexPair<Index>(2, 1)})).eval().shuffle(array<Index, 4>({0, 2, 1, 3})).reshape(array<Index, 3>({a.dimension(0) * b.dimension(0), a.dimension(1), b.dimension(2)})).stride(array<Index, 4>({a.dimension(0) + 1, 1, 1}));
}

// Unfortunately Eigen::TensorBase is super fucked, so in order to work with
// expression templates we use this unfortunate overload that requires two extra
// temporaries. Matrix multiplication is expensive though, so maybe this is
// more efficient anyways.
template <typename Derived1, typename Derived2>
inline Tensor<typename Derived1::Scalar, 3> matmul(const TensorBase<Derived1, ReadOnlyAccessors>& a, const TensorBase<Derived2, ReadOnlyAccessors>& b) {
  const Tensor<typename Derived1::Scalar, 3> c = static_cast<const Derived1&>(a);
  const Tensor<typename Derived2::Scalar, 3> d = static_cast<const Derived2&>(b);
  return c.contract(d, array<IndexPair<Index>, 1>({IndexPair<Index>(2, 1)})).eval().shuffle(array<Index, 4>({0, 2, 1, 3})).reshape(array<Index, 3>({c.dimension(0) * d.dimension(0), c.dimension(1), d.dimension(2)})).stride(array<Index, 4>({c.dimension(0) + 1, 1, 1}));
}

// Returns indices where x(i) is non zero.
template <typename Derived>
// inline Array<Index, Dynamic, 1> nonzero(const ArrayBase<Derived>& x) {
inline auto nonzero(const ArrayBase<Derived>& x) {
  Array<Index, Dynamic, 1> I = Array<Index, Dynamic, 1>::LinSpaced(x.size(), 0, x.size()-1);
  I.conservativeResize(std::stable_partition(I.data(), I.data() + I.size(), [&x](Index i){return x(i);}) - I.data());
  return I;
};

// Returns indices where x(i) is between min and max
template <typename Derived, typename Real>
inline auto between(const ArrayBase<Derived>& x, const Real& min, const Real& max, int inclusive = 0) {
  Array<Index, Dynamic, 1> I = Array<Index, Dynamic, 1>::LinSpaced(x.size(), 0, x.size() - 1);
  switch (inclusive) {
    case 0:
      I.conservativeResize(std::stable_partition(I.data(), I.data() + I.size(), [&x, &min, &max](int i){return ((min < x(i)) && (x(i) < max));}) - I.data());
      break;
    case 1:
      I.conservativeResize(std::stable_partition(I.data(), I.data() + I.size(), [&x, &min, &max](int i){return ((min <= x(i)) && (x(i) < max));}) - I.data());
      break;
    case 2:
      I.conservativeResize(std::stable_partition(I.data(), I.data() + I.size(), [&x, &min, &max](int i){return ((min < x(i)) && (x(i) <= max));}) - I.data());
      break;
    case 3:
      I.conservativeResize(std::stable_partition(I.data(), I.data() + I.size(), [&x, &min, &max](int i){return ((min <= x(i)) && (x(i) <= max));}) - I.data());
      break;
    default:
      throw std::invalid_argument("inclusive must be either 0, 1, 2, or 3 (not inclusive, inclusive only at lower bound, inclusive only at upper bound, inclusive at both bounds, respectively).");
  }
  return I;
}

// Returns indices where x(i) is greater than min
template <typename Derived, typename Real>
inline auto greater(const ArrayBase<Derived>& x, const Real& min, int inclusive = 0) {
  Array<Index, Dynamic, 1> I = Array<Index, Dynamic, 1>::LinSpaced(x.size(), 0, x.size() - 1);
  if (inclusive == 0) {
    I.conservativeResize(std::stable_partition(I.data(), I.data() + I.size(), [&x, &min](int i){return (min < x(i));}) - I.data());
  }
  else if (inclusive == 1) {
    I.conservativeResize(std::stable_partition(I.data(), I.data() + I.size(), [&x, &min](int i){return (min <= x(i));}) - I.data());
  }
  else {
    throw std::invalid_argument("inclusive must be either 0 or 1 (not inclusive or inclusive, respectively).");
  }
  return I;
}

// Returns indices where x(i) is less than max
template <typename Derived, typename Real>
inline auto less(const ArrayBase<Derived>& x, const Real& max, int inclusive = 0) {
  Array<Index, Dynamic, 1> I = Array<Index, Dynamic, 1>::LinSpaced(x.size(), 0, x.size() - 1);
  if (inclusive == 0) {
    I.conservativeResize(std::stable_partition(I.data(), I.data() + I.size(), [&x, &max](int i){return (x(i) < max);}) - I.data());
  }
  else if (inclusive == 1) {
    I.conservativeResize(std::stable_partition(I.data(), I.data() + I.size(), [&x, &max](int i){return (x(i) <= max);}) - I.data());
  }
  else {
    throw std::invalid_argument("inclusive must be either 0 or 1 (not inclusive or inclusive, respectively).");
  }
  return I;
}

// Returns difference between elements of a 1D array or vector
template <typename Derived>
inline auto diff(const DenseBase<Derived>& x) {
  static_assert(Derived::RowsAtCompileTime == 1 || Derived::ColsAtCompileTime == 1);
  if constexpr (Derived::SizeAtCompileTime = Dynamic) {
    return x.tail(x.size() - 1) - x.head(x.size() - 1);
  }
  else {
    return x.template tail<Derived::SizeAtCompileTime - 1>() - x.template head<Derived::SizeAtCompileTime - 1>();
  }
}

// convenient compile time function for 3^x
template <typename Real>
constexpr Real three_raise_x(Real x) {
  int p = (int) x;
  Real arg = (x - p) * ln_three<Real>();
  Real ret = 1;
  while (p > 0) {
    ret *= 3;
    p -= 1;
  }
  Real num = 1 + (arg / 2) + (arg * arg / 9) + (arg * arg * arg / 72) + (arg * arg * arg * arg / 1008) + (arg * arg * arg * arg * arg / 30240);
  Real den = 1 - (arg / 2) + (arg * arg / 9) - (arg * arg * arg / 72) + (arg * arg * arg * arg / 1008) - (arg * arg * arg * arg * arg / 30240);
  ret = ret * num / den;
  return ret;
}

} // tools
} // collocation

#endif // COLLOCATION_TOOLS_HPP
