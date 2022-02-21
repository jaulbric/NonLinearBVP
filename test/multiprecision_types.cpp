#include <boost/multiprecision/float128.hpp>
#include <iostream>
#include <iomanip>
#include <limits>

int main() {
  using boost::multiprecision::float128;

  float128 x = 5.209048393020202e-272Q;

  std::cout << std::numeric_limits<float128>::max() << std::endl;

  return 0;
}
