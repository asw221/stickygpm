
#define _USE_MATH_DEFINES
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>


#include <iostream>

#ifndef _EXTRA_DISTRIBUTIONS_
#define _EXTRA_DISTRIBUTIONS_


namespace extra_distributions {

  template< class RealType = double >
  RealType std_normal_cdf(const RealType &quant);

  // Algorithm from Wichura (1988) "The Percentage Points of the Normal
  // Distribution". JRSS C.
  template< class RealType = double >
  RealType std_normal_quantile(const RealType &prob);

  template< class RealType = double >
  RealType quantile(
    const std::normal_distribution<RealType> &d,
    const RealType &prob
  );



  template< class RealType = double >
  RealType std_logistic_cdf(const RealType& x);

  template< class RealType = double >
  RealType logistic_cdf(
    const RealType& x,
    const RealType& location = 0,
    const RealType& scale = 1
  );

  template< class RealType = double >
  RealType std_logistic_quantile(const RealType& prob);

  template< class RealType = double >
  RealType logistic_quantile(
    const RealType& prob,
    const RealType& location = 0,
    const RealType& scale = 1
  );
  
};




// -----------------------------------------------------------------------------





template< class RealType >
RealType extra_distributions::quantile(
  const std::normal_distribution<RealType> &d,
  const RealType &prob
) {
  if (prob <= 0.0 || prob >= 1.0)
    throw std::logic_error("Normal quantile out of range");
  const RealType q = std_normal_quantile(prob);
  return q * d.stddev() + d.mean();
};





template< class RealType >
RealType extra_distributions::std_normal_cdf(const RealType &quant) {
  const RealType p = 0.5 * std::erfc(-quant * M_SQRT1_2);
  return p;
};



// Algorithm from Wichura (1988) "The Percentage Points of the Normal
// Distribution". JRSS C.
template< class RealType >
RealType extra_distributions::std_normal_quantile(const RealType &prob) {
  static const RealType _SPLIT_Q_ = 0.425;
  static const RealType _SPLIT_R_ = 5.0;
  static const RealType _SPLIT_Q_SQ_ = 0.180625;
  static const RealType _WICHURA_CONST_2_ = 1.6;
  
  static const std::vector<RealType> _WICHURA_A_{
    3.3871328727963996080e0, 1.3314166789178437745e2, 1.9715909503065514427e3,
      1.3731693765509461125e4, 4.5921953931549871457e4, 6.7265770927008700853e4,
      3.3430575583588128105e4, 2.5090809287301226727e3 };

  static const std::vector<RealType> _WICHURA_B_{
    1.0, 4.2313330701600911252e1, 6.8718700749205790830e2,
      5.3941960214247511077e3, 2.1213794301586595867e4, 3.9307895800092710610e4,
      2.8729085735721942674e4, 5.2264952788528545610e3 };

  static const std::vector<RealType> _WICHURA_C_{
    1.42343711074968357734e0, 4.63033784615654529590e0, 5.76949722146069140550e0,
      3.64784832476320460504e0, 1.27045825245236838258e0, 2.41780725177450611770e-1,
      2.27238449892691845833e-2, 7.74545014278341407640e-4 };

  static const std::vector<RealType> _WICHURA_D_{
    1.0, 2.05319162663775882187e0, 1.67638483018380384940e0,
      6.89767334985100004550e-1, 1.48103976427480074590e-1, 1.51986665636164571966e-2,
      5.47593808499534494600e-4, 1.05075007164441684324e-9 };

  static const std::vector<RealType> _WICHURA_E_{
    6.65790464350110377720e0, 5.46378491116411436990e0, 1.78482653991729133580e0,
      2.96560571828504891230e-1, 2.65321895265761230930e-2, 1.24266094738807843860e-3,
      2.71155556874348757815e-5, 2.01033439929228813265e-7 };

  static const std::vector<RealType> _WICHURA_F_{
    1.0, 5.99832206555887937690e-1, 1.36929880922735805310e-1,
      1.48753612908506148525e-2, 7.86869131145613259100e-4, 1.84631831751005468180e-5,
      1.42151175831644588870e-7, 2.04426310338993978564e-15 };

  
  const RealType q = prob - 0.5;
  
  typename std::vector<RealType>::const_reverse_iterator nit, dit;
  RealType quant, r, numer, denom;
  
  if (std::abs(q) <= _SPLIT_Q_) {
    r = _SPLIT_Q_SQ_ - q * q;
    numer = _WICHURA_A_.back();
    denom = _WICHURA_B_.back();
    for (nit = _WICHURA_A_.rbegin() + 1, dit = _WICHURA_B_.rbegin() + 1;
	 nit != _WICHURA_A_.rend(); nit++, dit++) {
      numer = numer * r + (*nit);
      denom = denom * r + (*dit);
    }
    quant = q * numer / denom;
  }
  else {
    r = (q < 0.0) ? prob : (1.0 - prob);
    if (r <= 0.0)
      quant = std::numeric_limits<RealType>::max();
    else {
      r = std::sqrt(-std::log(r));
      if (r <= _SPLIT_R_) {
	r -= _WICHURA_CONST_2_;
	numer = _WICHURA_C_.back();
	denom = _WICHURA_D_.back();
	for (nit = _WICHURA_C_.rbegin() + 1, dit = _WICHURA_D_.rbegin() + 1;
	     nit != _WICHURA_C_.rend(); nit++, dit++) {
	  numer = numer * r + (*nit);
	  denom = denom * r + (*dit);
	}
	quant = numer / denom;
      }
      else {
	r -= _SPLIT_R_;
	numer = _WICHURA_E_.back();
	denom = _WICHURA_F_.back();
	for (nit = _WICHURA_E_.rbegin() + 1, dit = _WICHURA_F_.rbegin() + 1;
	     nit != _WICHURA_E_.rend(); nit++, dit++) {
	  numer = numer * r + (*nit);
	  denom = denom * r + (*dit);
	}
	quant = numer / denom;
      }  // if (r <= _SPLIT_R_) ... else ...
    }  // if (r <= 0.0) ... else ...
  }  // if (std::abs(q) <= _SPLIT_Q_) ... else ...

  quant = std::abs(quant) * ((q >= 0) ? 1 : -1);
  return quant;
};










template< class RealType >
RealType extra_distributions::std_logistic_cdf(const RealType& x) {
  return 1 / (1 + std::exp(-x));
};



template< class RealType >
RealType extra_distributions::logistic_cdf(
  const RealType& x,
  const RealType& location,
  const RealType& scale
) {
  if (scale <= 0) {
    throw std::domain_error("Logistic scale parameter must be > 0");
  }
  return extra_distributions::std_logistic_cdf((x - location)/scale);
};



template< class RealType >
RealType extra_distributions::std_logistic_quantile(
  const RealType& prob
) {
  RealType q;
  if ( prob < static_cast<RealType>( 0 ) ||
       prob > static_cast<RealType>( 1 ) ) {
    throw std::domain_error("Logistic inverse CDF defined on (0, 1)");
  }
  if ( prob == static_cast<RealType>( 0 ) ||
       prob == static_cast<RealType>( 1 ) ) {
#ifndef DNDEBUG
    std::cerr << "\t*** WARNING: std_logistic_quantile: Returning +/- Inf\n";
#endif
    q = 36.72;
    // ^^ If F(x) denotes the standard logistic CDF,
    // then F(36.72) = 0.9999999999999998
    if ( prob == static_cast<RealType>( 0 ) ) {
      q = -q;
    }
  }
  else {
    q = std::log(prob / (1 - prob));
  }
  return q;
};



template< class RealType >
RealType extra_distributions::logistic_quantile(
  const RealType& prob,
  const RealType& location,
  const RealType& scale
) {
  if ( prob <= static_cast<RealType>( 0 ) ||
       prob >= static_cast<RealType>( 1 ) ) {
    std::string msg =
      std::string("Logistic inverse CDF: argument was ") +
      std::to_string(prob) +
      std::string(" [defined on (0, 1)]");
    throw std::domain_error( msg );
  }
  if ( scale <= static_cast<RealType>( 0 ) ) {
    throw std::domain_error("Logistic scale parameter must be > 0");
  }
  return scale * std::log( prob / (1 - prob) ) + location;
};





#endif  // _EXTRA_DISTRIBUTIONS_
