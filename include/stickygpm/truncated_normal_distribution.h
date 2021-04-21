
#include <algorithm>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

#include "stickygpm/extra_distributions.h"


#ifndef _TRUNCATED_NORMAL_DISTRIBUTION_
#define _TRUNCATED_NORMAL_DISTRIBUTION_



template< class RealType >
class truncated_normal_distribution {
  
public:
  typedef RealType result_type;

  class param_type {
  private:
    RealType _high;       // upper bound
    RealType _low;        // lower bound
    RealType _mu;         // mean
    RealType _prob_high;  // CDF transform of _high
    RealType _prob_low;   // CDF transform of _low
    RealType _sigma;      // standard deviation

  public:
    using distribution_type = 
      truncated_normal_distribution<RealType>;

    param_type( const param_type& other );
    explicit param_type(
      RealType mean  = 0,
      RealType sd    = 1,
      RealType lower = 0,
      RealType upper = 1e4
    );

    RealType cdf_max() const;
    RealType cdf_min() const;
    RealType max()     const;
    RealType mean()    const;
    RealType min()     const;
    RealType stddev()  const;

    bool operator== ( const param_type& other ) const;
    bool operator!= ( const param_type& other ) const;

    template< class CharT, class Traits >
    friend std::basic_ostream<CharT, Traits>& operator<< (
      std::basic_ostream<CharT, Traits> &ost,
      const param_type &pt
    );
  };


  truncated_normal_distribution() :
    truncated_normal_distribution(0) { ; }
  
  explicit truncated_normal_distribution(
    RealType mean  = 0,
    RealType sd    = 1,
    RealType lower = 0,
    RealType upper = 1e4
  );
  
  explicit truncated_normal_distribution( const param_type &par );

  template< class Generator >
  RealType operator() ( Generator &g );

  template< class Generator >
  RealType operator() ( Generator &g, const param_type& params );

  RealType   max()    const;
  RealType   mean()   const;
  RealType   min()    const;
  RealType   stddev() const;
  param_type param()  const;

  void param( const param_type &par );
  void reset();

  bool operator== (
    const truncated_normal_distribution<RealType>& other
  ) const;
  
  bool operator!= (
    const truncated_normal_distribution<RealType>& other
  ) const;

  template< class CharT, class Traits >
  friend std::basic_ostream<CharT, Traits>& operator<< (
    std::basic_ostream<CharT, Traits> &ost,
    const truncated_normal_distribution<RealType> &d
  );


  
private:
  static const RealType _quantile_pivot;
  
  param_type _par;
  std::uniform_real_distribution<double> _Uniform_;

  template< class Generator >
  RealType _inverse_cdf( Generator &g, const param_type& params );
  
  template< class Generator >
  RealType _reject_rayleigh(
    Generator &g,
    const RealType Lo,
    const RealType Up
  );
};
// end - class truncated_normal_distribution





// --- Constants -----------------------------------------------------

template< class RealType >
const RealType truncated_normal_distribution<RealType>
::_quantile_pivot = 4.2648908;


// --- Constructors & Destructors ------------------------------------


template< class RealType >
truncated_normal_distribution<RealType>::param_type::param_type(
  const typename truncated_normal_distribution<RealType>
  ::param_type& other
) {
  _high = other._high;
  _low  = other._low;
  _mu   = other._mu;
  _prob_high = other._prob_high;
  _prob_low  = other._prob_low;
  _sigma     = other._sigma;
};


template< class RealType >
truncated_normal_distribution<RealType>::param_type::param_type(
  RealType mean, RealType sd,
  RealType lower, RealType upper
) {
  if (sd <= 0) {
    throw std::domain_error(
      "Normal distribution variance must be > 0");
  }
  if (lower == upper) {
    throw std::domain_error("Lower and upper bounds identical");
  }
  if (lower > upper) {
    std::cerr << "Truncated normal distribution: swapping bounds.\n"
	      << "(Lower bound greater than upper bound)\n";
    RealType temp = lower;
    lower = upper;
    upper = temp;
  }
  _high = upper;
  _low = lower;
  _mu = mean;
  _prob_high = std::min(
    static_cast<RealType>(1),
    extra_distributions::std_normal_cdf( (upper - mean) / sd )
  );
  _prob_low = std::max(
    static_cast<RealType>(0),
    extra_distributions::std_normal_cdf( (lower - mean) / sd )
  );
  _sigma = sd;
};


template< class RealType >
truncated_normal_distribution<RealType>
::truncated_normal_distribution(
  RealType mean, RealType sd, RealType lower, RealType upper
) {
  param(param_type(mean, sd, lower, upper));
};


template< class RealType >
truncated_normal_distribution<RealType>
::truncated_normal_distribution(
  const param_type &par
) {
  param(par);
};



// --- Utility Functions/Operators -----------------------------------


template< class RealType >
template< class Generator >
RealType truncated_normal_distribution<RealType>::_inverse_cdf(
  Generator &g,
  const typename truncated_normal_distribution<RealType>
    ::param_type& params
) {
  const RealType x = static_cast<RealType>( _Uniform_(g) ) *
    (params.cdf_max() - params.cdf_min()) + params.cdf_min();
  return extra_distributions::std_normal_quantile(x) *
    params.stddev() + params.mean();
};



template< class RealType >
template< class Generator >
RealType truncated_normal_distribution<RealType>::_reject_rayleigh(
  Generator &g,
  const RealType Lo,
  const RealType Up
) {
  const RealType c = Lo * Lo / 2;
  const RealType q = 1 -
    std::exp( c - Up * Up / 2 );
  RealType u, v, x;
  do {
    u = _Uniform_(g);
    v = _Uniform_(g);
    x = c - std::log( 1 - q * u );
  }
  while ( (v * v * x) > Lo );
  return std::sqrt( 2 * x );
};



template< class RealType >
template< class Generator >
RealType truncated_normal_distribution<RealType>::operator() (
  Generator &g
) {
  return operator() ( g, _par );
};


template< class RealType >
template< class Generator >
RealType truncated_normal_distribution<RealType>::operator() (
  Generator &g,
  const typename truncated_normal_distribution<RealType>
  ::param_type& params
) {
  const RealType L = (params.min() - params.mean()) / params.stddev();
  const RealType U = (params.max() - params.mean()) / params.stddev();
  if ( L >= _quantile_pivot ) {
    return _reject_rayleigh(g, L, U) *
      params.stddev() + params.mean();
  }
  else if ( U <= -_quantile_pivot ) {
    return -_reject_rayleigh(g, -U, -L) *
      params.stddev() + params.mean();
  }
  else {
    return _inverse_cdf( g, params );
  }
  // if ( r > _params.max() ) {
  //   std::cout << "Min: " << params.min() << "\n"
  // 	      << "Max: " << params.max() << "\n"
  // 	      << "Min CDF: " << params.cdf_min() << "\n"
  // 	      << "Max CDF: " << params.cdf_max() << "\n"
  // 	      << "p: " << x << "\n"
  // 	      << "Q: " << extra_distributions::std_normal_quantile(x) << "\n"
  // 	      << std::endl;
  // }
  return params.min();  // Not reached
};





template< class RealType >
bool truncated_normal_distribution<RealType>::param_type
::operator== (
  const typename
  truncated_normal_distribution<RealType>::param_type& other
) const {
  const bool paramsSame = (_mu == other._mu) &&
    (_sigma == other._sigma);
  const bool truncationSame = (_low == other._low) &&
    (_high == other._high);
  return paramsSame && truncationSame;
};


template< class RealType >
bool truncated_normal_distribution<RealType>::param_type
::operator!= (
  const typename
  truncated_normal_distribution<RealType>::param_type& other
) const {
  return !(*this == other);
};



template< class RealType >
bool truncated_normal_distribution<RealType>::operator== (
  const truncated_normal_distribution<RealType>& other
) const {
  return _par == other._par;
};



template< class RealType >
bool truncated_normal_distribution<RealType>::operator!= (
  const truncated_normal_distribution<RealType>& other
) const {
  return !(*this == other);
};




// --- Friend Functions ----------------------------------------------


template< class RealType, class CharT, class Traits >
std::basic_ostream<CharT, Traits>& operator<<(
  std::basic_ostream<CharT, Traits> &ost,
  const typename
  truncated_normal_distribution<RealType>::param_type &pt
) {
  ost << pt._mu << ", " << pt._sigma
      << "; [" << pt._low << ", " << pt._high << ")";
  return ost;
};



template< class RealType, class CharT, class Traits >
std::basic_ostream<CharT, Traits>& operator<<(
  std::basic_ostream<CharT, Traits> &ost,
  const truncated_normal_distribution<RealType> &d
) {
  ost << "Trunc-Normal(" << d.param() << ")\n";
  return ost;
};



// --- Getters -------------------------------------------------------

template< class RealType >
RealType truncated_normal_distribution<RealType>::param_type
::cdf_max() const {
  return _prob_high;
};


template< class RealType >
RealType truncated_normal_distribution<RealType>::param_type
::cdf_min() const {
  return _prob_low;
};


template< class RealType >
RealType truncated_normal_distribution<RealType>::param_type
::max() const {
  return _high;
};


template< class RealType >
RealType truncated_normal_distribution<RealType>::max() const {
  return _par.max();
};


template< class RealType >
RealType truncated_normal_distribution<RealType>::param_type
::mean() const {
  return _mu;
};


template< class RealType >
RealType truncated_normal_distribution<RealType>::mean() const {
  return _par.mean();
};


template< class RealType >
RealType truncated_normal_distribution<RealType>::param_type
::min() const {
  return _low;
};


template< class RealType >
RealType truncated_normal_distribution<RealType>::min() const {
  return _par.min();
};


template< class RealType >
RealType truncated_normal_distribution<RealType>::param_type
::stddev() const {
  return _sigma;
};


template< class RealType >
RealType truncated_normal_distribution<RealType>::stddev() const {
  return _par.stddev();
};


template< class RealType >
typename truncated_normal_distribution<RealType>::param_type
truncated_normal_distribution<RealType>::param() const {
  return _par;
};




// --- Setters -------------------------------------------------------

template< class RealType >
void truncated_normal_distribution<RealType>::param(
  const typename
  truncated_normal_distribution<RealType>::param_type &par
) {
  _par = par;
};


template< class RealType >
void truncated_normal_distribution<RealType>::reset() {
  _Uniform_.reset();
};


#endif  // _TRUNCATED_NORMAL_DISTRIBUTION_



