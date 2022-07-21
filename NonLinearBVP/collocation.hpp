#ifndef COLLOCATION_HPP
#define COLLOCATION_HPP

#ifndef COLLOCATION_VERBOSITY
#define COLLOCATION_VERBOSITY 0
#endif

#ifdef COLLOCATION_ALL_RETURN
#define COLLOCATION_FUNCTION_RETURN
#define COLLOCATION_BC_RETURN
#define COLLOCATION_FUNCTION_JACOBIAN_RETURN
#define COLLOCATION_BC_JACOBIAN_RETURN
#endif

// uncomment the following line to use Boost multiprecision types as Eigen scalar types
// #include <boost/multiprecision/eigen.hpp>

#include "NonLinearBVP/collocation/bvp4c.hpp"
#include "NonLinearBVP/collocation/bvp6c.hpp"

#endif // COLLOCATION_HPP
