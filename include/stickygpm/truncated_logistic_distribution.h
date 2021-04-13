
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

#include "stickygpm/extra_distributions.h"



#ifndef _TRUNCATED_LOGISTIC_DISTRIBUTION_
#define _TRUNCATED_LOGISTIC_DISTRIBUTION_




template< class RealType >
class truncated_logistic_distribution {
  
public:
  typedef RealType result_type;

  class param_type {
  private:
    RealType _high;       // upper bound
    RealType _low;        // lower bound
    RealType _mu;         // location
    RealType _prob_high;  // CDF transform of _high
    RealType _prob_low;   // CDF transform of _low
    RealType _sigma;      // standard deviation

  public:
    using distribution_type =
      truncated_logistic_distribution<RealType>;

    param_type( const param_type& other );
    explicit param_type(
      RealType location = 0,
      RealType scale = 1,
      RealType lower = 0,
      RealType upper = 1e4
    );

    RealType cdf_max()  const;
    RealType cdf_min()  const;
    RealType max()      const;
    RealType location() const;
    RealType min()      const;
    RealType scale()    const;

    bool operator== ( const param_type& other ) const;
    bool operator!= ( const param_type &other ) const;

    template< class CharT, class Traits >
    friend std::basic_ostream<CharT, Traits>& operator<< (
      std::basic_ostream<CharT, Traits> &ost,
      const param_type &pt
    );
  };
  // end class param_type


  truncated_logistic_distribution() :
    truncated_logistic_distribution(0) { ; }
  
  explicit truncated_logistic_distribution(
    RealType location = 0,
    RealType scale = 1,
    RealType lower = 0,
    RealType upper = 1e4
  );
  
  explicit truncated_logistic_distribution( const param_type &par );

  template< class Generator >
  RealType operator() (Generator &g);

  template< class Generator >
  RealType operator() ( Generator &g, const param_type& params );

  RealType   max()      const;
  RealType   location() const;
  RealType   min()      const;
  RealType   scale()    const;
  param_type param()    const;

  void param(const param_type &par);
  void reset();

  bool operator== (
    const truncated_logistic_distribution<RealType>& other
  ) const;
  
  bool operator!= (
    const truncated_logistic_distribution<RealType>& other
  ) const;

  template< class CharT, class Traits >
  friend std::basic_ostream<CharT, Traits>& operator<< (
    std::basic_ostream<CharT, Traits> &ost,
    const truncated_logistic_distribution<RealType> &d
  );


  
private:  
  param_type _par;
  std::uniform_real_distribution<double> _Uniform_;
};
// end - class truncated_logistic_distribution






// --- Constructors & Destructors ------------------------------------



template< class RealType >
truncated_logistic_distribution<RealType>::param_type::param_type(
  const typename truncated_logistic_distribution<RealType>
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
truncated_logistic_distribution<RealType>::param_type::param_type(
  RealType location,
  RealType scale,
  RealType lower,
  RealType upper
) {
  if (scale <= 0)
    throw std::logic_error(
      "Logistic distribution variance must be > 0");
  if (lower == upper)
    throw std::logic_error("Lower and upper bounds identical");
  if (lower > upper) {
    std::cerr << "Truncated logistic distribution: swapping "
	      << "bounds.\n"
	      << "(Lower bound greater than upper bound)\n";
    RealType temp = lower;
    lower = upper;
    upper = temp;
  }
  _high = upper;
  _low = lower;
  _mu = location;
  _prob_high = std::min(
    (result_type)1.0,
    extra_distributions::std_logistic_cdf(
      (upper - location) / scale )
  );
  _prob_low = std::max(
    (result_type)0.0,
    extra_distributions::std_logistic_cdf(
      (lower - location) / scale )
  );
  _sigma = scale;
};


template< class RealType >
truncated_logistic_distribution
<RealType>::truncated_logistic_distribution(
  RealType location,
  RealType scale,
  RealType lower,
  RealType upper
) {
  param( param_type(location, scale, lower, upper) );
};


template< class RealType >
truncated_logistic_distribution
<RealType>::truncated_logistic_distribution(
  const param_type &par
) {
  param(par);
};



// --- Utility Functions/Operators ---------------------------------------

template< class RealType >
template< class Generator >
RealType truncated_logistic_distribution<RealType>::operator() (
  Generator &g
) {
  return operator() ( g, _par );
};



template< class RealType >
template< class Generator >
RealType truncated_logistic_distribution<RealType>::operator() (
  Generator &g,
  const typename truncated_logistic_distribution<RealType>
  ::param_type& params
) {
  const RealType x = static_cast<RealType>( _Uniform_(g) ) *
    (params.cdf_max() - params.cdf_min()) + params.cdf_min();
  const RealType p = std::max(
    static_cast<RealType>(0),
    std::min( static_cast<RealType>(1), x )
  );
  return extra_distributions::std_logistic_quantile(p) * params.scale() +
    params.location();  
};



// --- Comparators ---------------------------------------------------


template< class RealType >
bool truncated_logistic_distribution<RealType>::param_type
::operator== (
  const typename truncated_logistic_distribution
    <RealType>::param_type& other
) const {
  const bool paramsSame = (_mu == other._mu) &&
    (_sigma == other._sigma);
  const bool truncationSame = (_low == other._low) &&
    (_high == other._high);
  return paramsSame && truncationSame;
};


template< class RealType >
bool truncated_logistic_distribution<RealType>::param_type
::operator!= (
  const typename truncated_logistic_distribution
    <RealType>::param_type& other
) const {
  return !(*this == other);
};



template< class RealType >
bool truncated_logistic_distribution<RealType>::operator== (
  const truncated_logistic_distribution<RealType>& other
) const {
  return _par == other._par;
};



template< class RealType >
bool truncated_logistic_distribution<RealType>::operator!= (
  const truncated_logistic_distribution<RealType>& other
) const {
  return !(*this == other);
};





// --- Friend Functions ----------------------------------------------


template< class RealType, class CharT, class Traits >
std::basic_ostream<CharT, Traits>& operator<<(
  std::basic_ostream<CharT, Traits> &ost,
  const typename truncated_logistic_distribution
    <RealType>::param_type& pt
) {
  ost << pt._mu << ", " << pt._sigma
      << "; [" << pt._low << ", " << pt._high << ")";
  return ost;
};



template< class RealType, class CharT, class Traits >
std::basic_ostream<CharT, Traits>& operator<<(
  std::basic_ostream<CharT, Traits> &ost,
  const truncated_logistic_distribution<RealType> &d
) {
  ost << "Trunc-Logistic(" << d.param() << ")\n";
  return ost;
};



// --- Getters -------------------------------------------------------

template< class RealType >
RealType
truncated_logistic_distribution<RealType>::param_type::cdf_max()
  const {
  return _prob_high;
};


template< class RealType >
RealType
truncated_logistic_distribution<RealType>::param_type::cdf_min()
  const {
  return _prob_low;
};


template< class RealType >
RealType
truncated_logistic_distribution<RealType>::param_type::max()
  const {
  return _high;
};


template< class RealType >
RealType
truncated_logistic_distribution<RealType>::max()
  const {
  return _par.max();
};


template< class RealType >
RealType
truncated_logistic_distribution<RealType>::param_type::location()
  const {
  return _mu;
};


template< class RealType >
RealType
truncated_logistic_distribution<RealType>::location()
  const {
  return _par.location();
};


template< class RealType >
RealType
truncated_logistic_distribution<RealType>::param_type::min()
  const {
  return _low;
};


template< class RealType >
RealType
truncated_logistic_distribution<RealType>::min()
  const {
  return _par.min();
};


template< class RealType >
RealType
truncated_logistic_distribution<RealType>::param_type::scale()
  const {
  return _sigma;
};


template< class RealType >
RealType
truncated_logistic_distribution<RealType>::scale()
  const {
  return _par.scale();
};


template< class RealType >
typename truncated_logistic_distribution<RealType>::param_type
truncated_logistic_distribution<RealType>::param() const {
  return _par;
};




// --- Setters -------------------------------------------------------

template< class RealType >
void truncated_logistic_distribution<RealType>::param(
  const typename
    truncated_logistic_distribution<RealType>::param_type &par
) {
  _par = par;
};


template< class RealType >
void truncated_logistic_distribution<RealType>::reset() {
  _Uniform_.reset();
};


#endif  // _TRUNCATED_LOGISTIC_DISTRIBUTION_




