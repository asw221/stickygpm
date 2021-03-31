
#include <cmath>
#include <Eigen/Core>
#include <stdexcept>



#ifndef _STICKYGPM_VECTOR_SUMMATION_
#define _STICKYGPM_VECTOR_SUMMATION_


namespace stickygpm {


  template< typename Derived, int SizeType0 >
  double vsum(const Eigen::Matrix<Derived, SizeType0, 1>& v) {
    double _A, _B, _correct = 0, _sum = 0;
    for ( int i = 0; i < v.size(); i++ ) {
      _A = static_cast<double>(v.coeffRef(i)) - _correct;
      _B = _sum + _A;
      _correct = (_B - _sum) - _A;
      _sum = _B;
    }
    return _sum;
  };


  // sum[t] = sum[t-1] + x[t] - {b[t-1] - sum[t-1] - a[t-1]}
  //
  // a[0] = x[0] ; b[0] = x[0]
  // err[0] = x[0] - 0 - x[0]
  // S[0] = x[0]
  //
  // a[1] = x[1] - {x[0] - 0 - x[0]}
  // b[1] = x[0] + x[1] - {x[0] - 0 - x[0]}
  // err[1] = {x[0] + x[1] - {x[0] - 0 - x[0]}} - x[0] - {x[1] - {x[0] - 0 - x[0]}}
  //        = {x[0] - x[0]} + {x[1] - x[1]}
  // S[1] = x[0] + x[1] - {x[0] - 0 - x[0]}

  

  template< typename Derived, int SizeType0v, int SizeType0w >
  double vdot(
    const Eigen::Matrix<Derived, SizeType0v, 1>& v,
    const Eigen::Matrix<Derived, SizeType0w, 1>& w
  ) {
#ifndef DNDEBUG
    if (v.size() != w.size()) {
      throw std::domain_error(
        "vdot: you mixed vectors of different lengths");
    }
#endif
    double _A, _B, _correct(0), _sum(0);
    for (int i = 0; i < v.size(); i++) {
      _A = static_cast<double>(v.coeffRef(i)) *
	static_cast<double>(w.coeffRef(i)) - _correct;
      _B = _sum + _A;
      _correct = (_B - _sum) - _A;
      _sum = _B;
    }
    return _sum;
  };


  
  template< typename Derived, int SizeType0 >
  double vdot(const Eigen::Matrix<Derived, SizeType0, 1>& v) {
    return stickygpm::vdot(v, v);
  };


  
  template< typename Derived, int SizeType0 >
  double v2norm(const Eigen::Matrix<Derived, SizeType0, 1>& v) {
    return std::sqrt(stickygpm::vdot(v, v));
  };



  template< typename Derived, int SizeType >
  double vsd(const Eigen::Matrix<Derived, SizeType, 1>& v) {
    const long int n = v.size();
    double sum_of_v = vsum(v);
    return std::sqrt( vdot(v) / n - sum_of_v * sum_of_v / (n * n) );
  };
  
}


#endif  // _STICKYGPM_VECTOR_SUMMATION_

