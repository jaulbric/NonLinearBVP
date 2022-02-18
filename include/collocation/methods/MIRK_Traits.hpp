#ifndef MIRK_TRAITS_HPP
#define MIRK_TRAITS_HPP

#include <Eigen/Dense>

using Eigen::Array;
using Eigen::ArrayBase;
using Eigen::Map;
using Eigen::Dynamic;
using Eigen::Index;

namespace collocation { namespace methods {

template<typename _Scalar, Index _RowsAtCompileTime, Index _ColsAtCompileTime, Index _ParamsAtCompileTime>
struct MIRK_Traits {
public:
  typedef _Scalar Scalar;
  static constexpr Index RowsAtCompileTime = _RowsAtCompileTime;
  static constexpr Index ColsAtCompileTime = _ColsAtCompileTime;
  static constexpr Index SizeAtCompileTime = _RowsAtCompileTime * _ColsAtCompileTime;
  static constexpr Index ParamsAtCompileTime = _ParamsAtCompileTime;
  static constexpr Index IntervalsAtCompileTime = _ColsAtCompileTime - 1;
  static constexpr Index ResiduesAtCompileTime = _RowsAtCompileTime * _ColsAtCompileTime + _ParamsAtCompileTime;
  static constexpr Index BCsAtCompileTime = _RowsAtCompileTime + _ParamsAtCompileTime;

protected:
  Array<Scalar, ParamsAtCompileTime, 1> m_p;

public:
  template <typename Derived>
  MIRK_Traits(const ArrayBase<Derived>& p_input) : m_p{p_input}, p{m_p} {};

  ~MIRK_Traits() = default;

  const Array<Scalar, ParamsAtCompileTime, 1>& p;
};

template<typename _Scalar, Index _RowsAtCompileTime, Index _ColsAtCompileTime>
struct MIRK_Traits<_Scalar, _RowsAtCompileTime, _ColsAtCompileTime, 0> {
  typedef _Scalar Scalar;
  static constexpr Index RowsAtCompileTime = _RowsAtCompileTime;
  static constexpr Index ColsAtCompileTime = _ColsAtCompileTime;
  static constexpr Index SizeAtCompileTime = _RowsAtCompileTime * _ColsAtCompileTime;
  static constexpr Index ParamsAtCompileTime = 0;
  static constexpr Index IntervalsAtCompileTime = _RowsAtCompileTime - 1;
  static constexpr Index ResiduesAtCompileTime = _RowsAtCompileTime * _ColsAtCompileTime;
  static constexpr Index BCsAtCompileTime = _RowsAtCompileTime;

  MIRK_Traits() = default;

  ~MIRK_Traits() = default;
};

template<typename _Scalar, Index _RowsAtCompileTime, Index _ColsAtCompileTime>
struct MIRK_Traits<_Scalar, _RowsAtCompileTime, _ColsAtCompileTime, Dynamic> {
public:
  typedef _Scalar Scalar;
  static constexpr Index RowsAtCompileTime = _RowsAtCompileTime;
  static constexpr Index ColsAtCompileTime = _ColsAtCompileTime;
  static constexpr Index SizeAtCompileTime = _RowsAtCompileTime * _ColsAtCompileTime;
  static constexpr Index ParamsAtCompileTime = Dynamic;
  static constexpr Index IntervalsAtCompileTime = _ColsAtCompileTime - 1;
  static constexpr Index ResiduesAtCompileTime = Dynamic;
  static constexpr Index BCsAtCompileTime = Dynamic;

protected:
  Array<Scalar, Dynamic, 1> m_p;

public:
  template <typename Derived>
  MIRK_Traits(const ArrayBase<Derived>& p_input) : m_p{p_input}, p{m_p} {};

  ~MIRK_Traits() = default;

  const Array<Scalar, Dynamic, 1>& p;
};

template<typename _Scalar, Index _RowsAtCompileTime, Index _ParamsAtCompileTime>
struct MIRK_Traits<_Scalar, _RowsAtCompileTime, Dynamic, _ParamsAtCompileTime> {
public:
  typedef _Scalar Scalar;
  static constexpr Index RowsAtCompileTime = _RowsAtCompileTime;
  static constexpr Index ColsAtCompileTime = Dynamic;
  static constexpr Index SizeAtCompileTime = Dynamic;
  static constexpr Index ParamsAtCompileTime = _ParamsAtCompileTime;
  static constexpr Index IntervalsAtCompileTime = Dynamic;
  static constexpr Index ResiduesAtCompileTime = Dynamic;
  static constexpr Index BCsAtCompileTime = _RowsAtCompileTime + _ParamsAtCompileTime;

protected:
  Array<Scalar, ParamsAtCompileTime, 1> m_p;

public:
  template <typename Derived>
  MIRK_Traits(const ArrayBase<Derived>& p_input) : m_p{p_input}, p{m_p} {};

  ~MIRK_Traits() = default;

  const Array<Scalar, ParamsAtCompileTime, 1>& p;
};

template<typename _Scalar, Index _RowsAtCompileTime>
struct MIRK_Traits<_Scalar, _RowsAtCompileTime, Dynamic, 0> {
public:
  typedef _Scalar Scalar;
  static constexpr Index RowsAtCompileTime = _RowsAtCompileTime;
  static constexpr Index ColsAtCompileTime = Dynamic;
  static constexpr Index SizeAtCompileTime = Dynamic;
  static constexpr Index ParamsAtCompileTime = 0;
  static constexpr Index IntervalsAtCompileTime = Dynamic;
  static constexpr Index ResiduesAtCompileTime = Dynamic;
  static constexpr Index BCsAtCompileTime = _RowsAtCompileTime;

  MIRK_Traits() = default;

  ~MIRK_Traits() = default;
};

template<typename _Scalar, Index _ColsAtCompileTime, Index _ParamsAtCompileTime>
struct MIRK_Traits<_Scalar, Dynamic, _ColsAtCompileTime, _ParamsAtCompileTime> {
public:
  typedef _Scalar Scalar;
  static constexpr Index RowsAtCompileTime = Dynamic;
  static constexpr Index ColsAtCompileTime = _ColsAtCompileTime;
  static constexpr Index SizeAtCompileTime = Dynamic;
  static constexpr Index ParamsAtCompileTime = _ParamsAtCompileTime;
  static constexpr Index IntervalsAtCompileTime = _ColsAtCompileTime - 1;
  static constexpr Index ResiduesAtCompileTime = Dynamic;
  static constexpr Index BCsAtCompileTime = Dynamic;

protected:
  Array<Scalar, ParamsAtCompileTime, 1> m_p;

public:
  template <typename Derived>
  MIRK_Traits(const ArrayBase<Derived>& p_input) : m_p{p_input}, p{m_p} {};

  ~MIRK_Traits() = default;

  const Array<Scalar, ParamsAtCompileTime, 1>& p;
};

template<typename _Scalar, Index _ColsAtCompileTime>
struct MIRK_Traits<_Scalar, Dynamic, _ColsAtCompileTime, 0> {
public:
  typedef _Scalar Scalar;
  static constexpr Index RowsAtCompileTime = Dynamic;
  static constexpr Index ColsAtCompileTime = _ColsAtCompileTime;
  static constexpr Index SizeAtCompileTime = Dynamic;
  static constexpr Index ParamsAtCompileTime = 0;
  static constexpr Index IntervalsAtCompileTime = _ColsAtCompileTime - 1;
  static constexpr Index ResiduesAtCompileTime = Dynamic;
  static constexpr Index BCsAtCompileTime = Dynamic;

  MIRK_Traits() = default;

  ~MIRK_Traits() = default;
};

template<typename _Scalar, Index _ParamsAtCompileTime>
struct MIRK_Traits<_Scalar, Dynamic, Dynamic, _ParamsAtCompileTime> {
public:
  typedef _Scalar Scalar;
  static constexpr Index RowsAtCompileTime = Dynamic;
  static constexpr Index ColsAtCompileTime = Dynamic;
  static constexpr Index SizeAtCompileTime = Dynamic;
  static constexpr Index ParamsAtCompileTime = _ParamsAtCompileTime;
  static constexpr Index IntervalsAtCompileTime = Dynamic;
  static constexpr Index ResiduesAtCompileTime = Dynamic;
  static constexpr Index BCsAtCompileTime = Dynamic;

protected:
  Array<Scalar, ParamsAtCompileTime, 1> m_p;

public:
  template <typename Derived>
  MIRK_Traits(const ArrayBase<Derived>& p_input) : m_p{p_input}, p{m_p} {};

  ~MIRK_Traits() = default;

  const Array<Scalar, ParamsAtCompileTime, 1>& p;
};

template<typename _Scalar>
struct MIRK_Traits<_Scalar, Dynamic, Dynamic, 0> {
public:
  typedef _Scalar Scalar;
  static constexpr Index RowsAtCompileTime = Dynamic;
  static constexpr Index ColsAtCompileTime = Dynamic;
  static constexpr Index SizeAtCompileTime = Dynamic;
  static constexpr Index ParamsAtCompileTime = 0;
  static constexpr Index IntervalsAtCompileTime = Dynamic;
  static constexpr Index ResiduesAtCompileTime = Dynamic;
  static constexpr Index BCsAtCompileTime = Dynamic;

  MIRK_Traits() = default;

  ~MIRK_Traits() = default;
};

} // namespace methods
} // namespace collocation

#endif // MIRK_TRAITS_HPP
