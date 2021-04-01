
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>



/*
 * Following:
 *
 *   Holmes & Held (2006) Bayesian Analysis.
 *   "Bayesian Auxiliary Variable Models for Binary and Multinomial 
 *    Regression"
 *   < Appendix: A4 >
 *
 */



#ifndef _LOGISTIC_WEIGHT_DISTRIBUTION_
#define _LOGISTIC_WEIGHT_DISTRIBUTION_


template< class RealType = double >
class logistic_weight_distribution;


template< class RealType >
bool operator== (
  const logistic_weight_distribution<RealType>& lhs,
  const logistic_weight_distribution<RealType>& rhs
);


template< class RealType >
bool operator!= (
  const logistic_weight_distribution<RealType>& lhs,
  const logistic_weight_distribution<RealType>& rhs
);




template< class RealType >
class logistic_weight_distribution {

public:
  typedef RealType result_type;

  class param_type {
  private:
    RealType _theta;
    
  public:
    typedef logistic_weight_distribution<RealType> distribution_type;

    explicit param_type(RealType theta = 1);
    RealType max() const;
    RealType min() const;
    RealType theta() const;

    friend bool operator== (const param_type& lhs, const param_type& rhs);
    friend bool operator!= (const param_type& lhs, const param_type& rhs);

    template< class CharT, class Traits >
    friend std::basic_ostream<CharT, Traits>& operator<< (
      std::basic_ostream<CharT, Traits>& ost,
      const param_type &pt
    );
  };


  explicit logistic_weight_distribution(RealType theta = 1);
  explicit logistic_weight_distribution(const param_type &par);

  template< class Generator >
  RealType operator() (Generator& g, const int maxIt = 100);

  RealType theta() const;
  param_type param() const;

  void param(const param_type &par);
  void reset();

  friend bool operator==<RealType> (
    const logistic_weight_distribution<RealType> &lhs,
    const logistic_weight_distribution<RealType> &rhs
  );
  
  friend bool operator!=<RealType> (
    const logistic_weight_distribution<RealType> &lhs,
    const logistic_weight_distribution<RealType> &rhs
  );

  template< class CharT, class Traits >
  friend std::basic_ostream<CharT, Traits>& operator<< (
    std::basic_ostream<CharT, Traits>& ost,
    const logistic_weight_distribution<RealType>& d
  );

  static const RealType _interval_pivot;
  static const RealType _eps_r;
  static const RealType _lambda_0;
  static const RealType _LN_PI;
  static const RealType _PI2;

private:
  param_type _par;
  std::uniform_real_distribution<RealType> _Uniform_;
  std::normal_distribution<RealType> _Gaussian_;

  bool _rightmost_interval(
    const RealType& U,
    const RealType& lam,
    const int maxIt = 100
  );
  bool _leftmost_interval(
    const RealType& U,
    const RealType& lam,
    const int maxIt = 100
  );
  
};






// --- Utility -------------------------------------------------------

template< class RealType >
const RealType logistic_weight_distribution<RealType>::_interval_pivot =
  4.0 / 3;


template< class RealType >
const RealType logistic_weight_distribution<RealType>::_eps_r = 1e-4;


template< class RealType >
const RealType logistic_weight_distribution<RealType>::_lambda_0 =
  1.6454212694466686;
// If the input residual is below _eps_r in magnitude, the return
// value is non-random numerically


template< class RealType >
const RealType logistic_weight_distribution<RealType>::_LN_PI =
  std::log(M_PI);


template< class RealType >
const RealType logistic_weight_distribution<RealType>::_PI2 =
  M_PI * M_PI;



template< class RealType >
template< class Generator >
RealType logistic_weight_distribution<RealType>::operator() (
  Generator& g,
  const int maxIt
) {
  const RealType r = std::sqrt( _par.theta() );
  if ( r <= _eps_r ) {
    return _lambda_0;
  }
  int iteration = 0;
  bool ok = false;
  RealType Y, U, lam;
  
  do {
    
    Y = _Gaussian_(g);
    U = _Uniform_(g);
    Y *= Y;
    Y = 1 + (Y - std::sqrt( Y * (4 * r + Y) )) / (2 * r);
    if ( U <= 1 / (1 + Y) )
      lam = r / Y;
    else
      lam = r * Y;
    // lam ~ Gen. Inv-Gaussian(0.5, 1, theta)

    if ( lam > _interval_pivot )
      ok = _rightmost_interval(U, lam);
    else
      ok = _leftmost_interval(U, lam);
    iteration++;
  
  }
  while ( !ok && iteration < maxIt );

  if ( !ok ) {
    std::cerr << "\nlogistic_weight_distribution: maxIt reached"
	      << " during sampling; r = " << r
	      << "; lambda = " << lam
	      << std::endl;
  }

  return lam;
};




template< class RealType >
bool logistic_weight_distribution<RealType>::_rightmost_interval(
  const RealType& U,
  const RealType& lam,
  const int maxIt
) {
  RealType Z = 1, X = std::exp(-0.5 * lam), j = 0, A;
  bool terminate_subroutine = false;
  bool ok = false;
  int iteration = 0;
  while ( !terminate_subroutine && iteration < maxIt ) {
    j++;
    A = (j + 1) * (j + 1);
    Z = Z - ( A * std::pow(X, A - 1) );
    if ( Z >= U ) {
      terminate_subroutine = true;
      ok = true;
    }
    else {
      j++;
      Z = Z - ( A * std::pow(X, A - 1) );
      if ( Z < U )
	terminate_subroutine = true;
    }
    iteration++;
  }
  return ok;
};
					    



template< class RealType >
bool logistic_weight_distribution<RealType>::_leftmost_interval(
  const RealType& U,
  const RealType& lam,
  const int maxIt
) {
  RealType X = -_PI2 / (2 * lam);
  const RealType H =
    0.5 * M_LN2
    + 2.5 * _LN_PI
    - 2.5 * std::log(lam)
    + X
    + 0.5 * lam;
  const RealType lnU = std::log(U), K = lam / _PI2;
  RealType Z = 1, j = 0, A;
  bool terminate_subroutine = false;
  bool ok = false;
  int iteration = 0;
  X = std::exp(X);
  while ( !terminate_subroutine && iteration < maxIt ) {
    j++;
    Z = Z - K * std::pow(X, j * j - 1);
    if ( H + std::log(Z) > lnU ) {
      terminate_subroutine = true;
      ok = true;
    }
    else {
      j++;
      A = (j + 1) * (j + 1);
      Z = Z + A * std::pow(X, A - 1);
      if ( H + std::log(Z) < lnU )
	terminate_subroutine = true;
    }
    iteration++;
  }
  return ok;
};


  


// --- Constructors & Destructors ------------------------------------

template< class RealType >
logistic_weight_distribution<RealType>::param_type::param_type(
  RealType theta
) {
  if ( theta < 0 ) {
    throw std::domain_error(
      "logistic_weight_distribution: parameter must be >= 0");
  }
  _theta = theta;
};


template< class RealType >
logistic_weight_distribution<RealType>::logistic_weight_distribution(
  RealType theta
) {
  param(param_type(theta));
};

template< class RealType >
logistic_weight_distribution<RealType>::logistic_weight_distribution(
  const typename logistic_weight_distribution<RealType>::param_type& par
) {
  param(par);
};


// --- Friend Functions ----------------------------------------------

template< class RealType >
bool operator== (
  const typename logistic_weight_distribution<RealType>::param_type& lhs,
  const typename logistic_weight_distribution<RealType>::param_type& rhs
) {
  return lhs._theta == rhs._theta;
};


template< class RealType >
bool operator!= (
  const typename logistic_weight_distribution<RealType>::param_type& lhs,
  const typename logistic_weight_distribution<RealType>::param_type& rhs
) {
  return !(lhs == rhs);
};


template< class RealType >
bool operator== (
  const logistic_weight_distribution<RealType>& lhs,
  const logistic_weight_distribution<RealType>& rhs
) {
  return lhs._par == rhs._par;
};

template< class RealType >
bool operator!= (
  const logistic_weight_distribution<RealType>& lhs,
  const logistic_weight_distribution<RealType>& rhs
) {
  return !(lhs == rhs);
};




template< class RealType, class CharT, class Traits >
std::basic_ostream<CharT, Traits>& operator<< (
  std::basic_ostream<CharT, Traits>& ost,
  const typename logistic_weight_distribution<RealType>::param_type& par
) {
  ost << par._theta;
  return ost;
};


template< class RealType, class CharT, class Traits >
std::basic_ostream<CharT, Traits>& operator<< (
  std::basic_ostream<CharT, Traits>& ost,
  const logistic_weight_distribution<RealType>& d
) {
  ost << "Logistic-Wt(" << d._par << ")\n";
  return ost;
};



// --- Getters -------------------------------------------------------

template< class RealType >
RealType logistic_weight_distribution<RealType>::param_type::max()
  const {
  return std::numeric_limits<RealType>::infinity();
};

template< class RealType >
RealType logistic_weight_distribution<RealType>::param_type::min()
  const {
  return 0;
};

template< class RealType >
RealType logistic_weight_distribution<RealType>::param_type::theta()
  const {
  return _theta;
};



template< class RealType >
RealType logistic_weight_distribution<RealType>::theta() const {
  return _par.theta();
};

template< class RealType >
typename logistic_weight_distribution<RealType>::param_type
logistic_weight_distribution<RealType>::param() const {
  return _par;
};


// --- Setters -------------------------------------------------------

template< class RealType >
void logistic_weight_distribution<RealType>::param(
  const typename logistic_weight_distribution<RealType>::param_type& par
) {
  _par = par;
};

template< class RealType >
void logistic_weight_distribution<RealType>::reset() {
  _Uniform_.reset();
  _Gaussian_.reset();
};


#endif  // _LOGISTIC_WEIGHT_DISTRIBUTION_

