
#define _USE_MATH_DEFINES
// #define __STDCPP_WANT_MATH_SPEC_FUNCS__

#include <boost/math/special_functions/bessel.hpp>
#include <cassert>
#include <cmath>
#include <iterator>  // std::distance
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>


#ifndef _STICKYGPM_COVARIANCE_FUNCTORS_
#define _STICKYGPM_COVARIANCE_FUNCTORS_


namespace stickygpm {

  // Put these somewhere else:
  template< typename FuncT, typename T = typename FuncT::result_type >
  T gradient_x( const FuncT& f, const T x, const T h = 1e-6 ) {
    assert( h > static_cast<T>(0) );
    return (f(x + h) - f(x - h)) / (2 * h);
  };

  template< typename FuncT, typename T = typename FuncT::result_type >
  T gradient_xf( const FuncT& f, const T x, const T h = 1e-6 ) {
    assert( h > static_cast<T>(0) );
    return (f(x + h) - f(x)) / h;
  };

  template< typename FuncT, typename T = typename FuncT::result_type >
  T gradient_xb( const FuncT& f, const T x, const T h = 1e-6 ) {
    assert( h > static_cast<T>(0) );
    return (f(x) - f(x - h)) / h;
  };

  
  
  
  template< typename T = double >
  class covariance_functor {
  public:
    typedef T result_type;

    class param_type {
    private:
      std::vector<T> _theta;
      
    public:
      param_type() { ; }
      param_type( const param_type& other );
      explicit param_type( const int n );
      
      template< typename InputIt >
      param_type( InputIt first, InputIt last );
      
      T& operator[]( int pos );
      const T& operator[]( int pos ) const;
      size_t size() const;

      friend std::ostream& operator<<(
        std::ostream& os,
	const param_type& param
      ) {
	os << "\u03B8" << " = (";
	for ( unsigned i = 0; i < param.size(); i++ ) {
	  os << param[i] << ", ";
	}
	os << "\b\b) ";
	return os;
      };
      // friend bool operator== (const param_type& lhs, const param_type& rhs);
    };
    // class param_type

    covariance_functor() { ; }
    covariance_functor( const covariance_functor<T>& other );
    explicit covariance_functor( const int n );
    
    template< typename InputIt >
    covariance_functor( InputIt first, InputIt last ) :
      _par( first, last )
    { ; }
    // Can put SFINAE here ^^
    //   typename std::enable_if_t<std::is_floating_point<T>::value, bool> = true

    virtual ~covariance_functor() { ; }
    
    virtual T operator() ( const T val ) const;
    virtual T inverse( const T cov ) const;
    virtual T fwhm() const;
    // friend bool operator== (const covariance_functor<T>& lhs, const covariance_functor<T>& rhs);

    virtual std::vector<T> gradient( const T val ) const;
    virtual std::vector<T> param_lower_bounds() const;
    virtual std::vector<T> param_upper_bounds() const;

    virtual void param( const param_type& par );
    param_type param() const;
    size_t param_size() const;

  protected:
    param_type _par;
  };
  // class covariance_functor



  template< typename T = double >
  class radial_basis :
    public covariance_functor<T> {
  public:
    typedef T result_type;
    using param_type = typename covariance_functor<T>::param_type;

    radial_basis() : covariance_functor<T>(3) { ; }
    
    template< typename InputIt >
    radial_basis( InputIt first, InputIt last );

    T operator() ( const T val ) const;
    T inverse( const T val ) const;
    T fwhm() const;

    std::vector<T> gradient( const T val ) const;
    std::vector<T> param_lower_bounds() const;
    std::vector<T> param_upper_bounds() const;

    T variance() const;
    T bandwidth() const;
    T exponent() const;

    void param( const param_type& par );
    void variance( const T val );
    void bandwidth( const T val );
    void exponent( const T val );

  private:
    void _validate_parameters() const;
  };



  /*
   * Matern covariance
   * 
   * Parameters are (sigma^2, nu, rho), all of which should be > 0
   *
   * C_nu(d) = sigma^2 * 2^(1 - nu) / Gamma(nu) *
   *             (sqrt(2 * nu) * d / rho)^nu * 
   *             K_nu( sqrt(2 * nu) * d / rho ),
   *
   * where K_nu(*) is the modified Bessel function of the second kind,
   * with order nu.
   *
   */
  template< typename T = double >
  class matern :
    public covariance_functor<T> {
  public:
    typedef T result_type;
    using param_type = typename covariance_functor<T>::param_type;

    matern();

    template< typename InputIt >
    matern( InputIt first, InputIt last );

    T operator() ( const T val ) const;
    T inverse( const T val ) const;
    T fwhm() const;

    std::vector<T> gradient( const T val ) const;
    std::vector<T> param_lower_bounds() const;
    std::vector<T> param_upper_bounds() const;

    T variance() const;
    T nu() const;
    T rho() const;
    
    void param( const param_type& par );
    void variance( const T val );
    void nu( const T val );
    void rho( const T val );

  private:
    static T _eps;
    static T _tol;
    static int _max_it;
    T _norm_c;
    T _sqrt_2nu_rho;
    void _compute_normalizing_constant();
  };

  
  
}
// namespace stickygpm


// --- covariance_functor<T>::param_type -----------------------------

template< typename T >
stickygpm::covariance_functor<T>::param_type::param_type(
  const param_type& other
) {
  _theta.assign( other._theta.cbegin(),
		 other._theta.cend() );
};


template< typename T >
template< typename InputIt >
stickygpm::covariance_functor<T>::param_type::param_type(
  InputIt first, InputIt last
) {
  _theta.assign( first, last );
};


template< typename T >
stickygpm::covariance_functor<T>::param_type::param_type(
  const int n
) {
  _theta.resize( n, 1 );
};



template< typename T >
T& stickygpm::covariance_functor<T>::param_type::operator[](
  int pos
) {
  return _theta[pos];
};

template< typename T >
const T& stickygpm::covariance_functor<T>::param_type::operator[](
  int pos
) const {
  return _theta[pos];
};


template< typename T >
size_t stickygpm::covariance_functor<T>::param_type::size() const {
  return _theta.size();
};



// template< typename T >
// std::ostream& stickygpm::operator<<(
//   std::ostream& os,
//   const typename stickygpm::covariance_functor<T>::param_type& param
// ) {
//   os << "\u03B8" << " = (";
//   for ( unsigned i = 0; i < param.size(); i++ ) {
//     os << param[i] << ", ";
//   }
//   os << "\b\b) ";
//   return os;
// };



// --- covariance_functor<T> -----------------------------------------

template< typename T >
stickygpm::covariance_functor<T>::covariance_functor(
  const covariance_functor<T>& other
) :
  _par(other._par)
{ ; }


template< typename T >
stickygpm::covariance_functor<T>::covariance_functor(
  const int n
) :
  _par( n )
{ ; }


template< typename T >
T stickygpm::covariance_functor<T>::operator() ( const T val ) const {
  return val;
};

template< typename T >
T stickygpm::covariance_functor<T>::inverse( const T cov ) const {
  return cov;
};


template< typename T >
T stickygpm::covariance_functor<T>::fwhm() const {
  return static_cast<T>( HUGE_VAL );
};


template< typename T >
std::vector<T>
stickygpm::covariance_functor<T>::gradient( const T val ) const {
  return std::vector<T>( _par.size(), 0 );
};


template< typename T >
std::vector<T>
stickygpm::covariance_functor<T>::param_lower_bounds() const {
  std::vector<T> bound{ 0, 0, 0 };
  return bound;
};


template< typename T >
std::vector<T>
stickygpm::covariance_functor<T>::param_upper_bounds() const {
  std::vector<T> bound( 3, static_cast<T>(HUGE_VAL) );
  return bound;
};



template< typename T >
void stickygpm::covariance_functor<T>::param(
  const stickygpm::covariance_functor<T>::param_type& par
) {
  _par = par;
};


template< typename T >
typename stickygpm::covariance_functor<T>::param_type
stickygpm::covariance_functor<T>::param() const {
  return _par;
};

template< typename T >
size_t stickygpm::covariance_functor<T>::param_size() const {
  return _par.size();
};



// --- radial_basis<T> -----------------------------------------------

template< typename T >
template< typename InputIt >
stickygpm::radial_basis<T>::radial_basis(
  InputIt first,
  InputIt last
) :
  covariance_functor<T>(first, last)
{
  if ( std::distance(first, last) != 3 ) {
    throw std::domain_error(
      "radial_basis functor requires 3 parameters");
  }
  _validate_parameters();
};




template< typename T >
T stickygpm::radial_basis<T>::operator() ( const T val ) const {
  return variance() *
    std::exp( -bandwidth() *
	      std::pow(std::abs(val), exponent()) );
};


template< typename T >
T stickygpm::radial_basis<T>::inverse( const T cov ) const {
  assert( cov > 0 && "radial_basis: argument to inverse must be > 0");
  return std::pow( -std::log(cov / variance()) / bandwidth(),
		   1 / exponent() );
};


template< typename T >
T stickygpm::radial_basis<T>::fwhm() const {
  return 2.0 * std::pow( std::log(2.0) / bandwidth(), 1 / exponent());
};



template< typename T >
std::vector<T>
stickygpm::radial_basis<T>::gradient( const T val ) const {
  const T c = operator()( val );
  std::vector<T> grad( 3 );
  grad[0] = c / variance();
  grad[1] = -std::pow( std::abs(val), exponent() ) * c;
  grad[2] = grad[1] * bandwidth() * std::log( std::abs(val) + 1e-8 );
  return grad;
};



template< typename T >
std::vector<T>
stickygpm::radial_basis<T>::param_lower_bounds() const {
  std::vector<T> bound{ 0, 0, 0 };
  return bound;
};


template< typename T >
std::vector<T>
stickygpm::radial_basis<T>::param_upper_bounds() const {
  const T huge = static_cast<T>( HUGE_VAL );
  std::vector<T> bound{ huge, huge, 2.0 };
  return bound;
};


// f(d) := tau^2 * exp( -psi |d|^nu )
// d/d(psi) f(d) = -|d|^nu * f(d)
// d/d(nu) f(d) = -psi * |d|^nu * log|d| * f(d)



template< typename T >
T stickygpm::radial_basis<T>::variance() const {
  return this->_par[0];
};

template< typename T >
T stickygpm::radial_basis<T>::bandwidth() const {
  return this->_par[1];
};

template< typename T >
T stickygpm::radial_basis<T>::exponent() const {
  return this->_par[2];
};



template< typename T >
void stickygpm::radial_basis<T>::param(
  const stickygpm::radial_basis<T>::param_type& par
) {
  assert( par.size() == 3 && "Invalid parameter size" );
  assert( par[0] > 0 && "Invalid parameter (0)");
  assert( par[1] > 0 && "Invalid parameter (1)");
  assert( par[2] > 0 && par[2] <= 2 && "Invalid parameter (2)");
  this->_par = par;
};


template< typename T >
void stickygpm::radial_basis<T>::variance( const T val ) {
  if ( val <= 0 ) {
    throw std::domain_error(
      "radial_basis functor: variance parameter"
      " must be > 0" );
  }
  this->_par[0] = val;
};

template< typename T >
void stickygpm::radial_basis<T>::bandwidth( const T val ) {
  if ( val <= 0 ) {
    throw std::domain_error(
      "radial_basis functor: bandwidth parameter"
      " must be > 0" );
  }
  this->_par[1] = val;
};

template< typename T >
void stickygpm::radial_basis<T>::exponent( const T val ) {
  if ( val <= 0 || val > 2 ) {
    throw std::domain_error(
      "radial_basis functor: exponent parameter"
      " must be on (0, 2]" );
  }
  this->_par[2] = val;
};


template< typename T >
void stickygpm::radial_basis<T>::_validate_parameters() const {
  if ( this->_par.size() != 3 ) {
    throw std::domain_error(
      "radial_basis functor: parameter must be size = 3" );
  }
  if ( this->_par[0] <= 0 ) {
    throw std::domain_error(
      "radial_basis functor: variance parameter"
      " must be > 0" );
  }
  if ( this->_par[1] <= 0 ) {
    throw std::domain_error(
      "radial_basis functor: bandwidth parameter"
      " must be > 0" );
  }
  if ( this->_par[2] <= 0 || this->_par[2] > 2 ) {
    throw std::domain_error(
      "radial_basis functor: exponent parameter"
      " must be on (0, 2]" );
  }
};




// --- matern<T> -----------------------------------------------------


template< typename T >
T stickygpm::matern<T>::_eps = 1e-6;

template< typename T >
T stickygpm::matern<T>::_tol = 1e-6;

template< typename T >
int stickygpm::matern<T>::_max_it = 100;



template< typename T >
stickygpm::matern<T>::matern() :
  covariance_functor<T>( 3 )
{
  _compute_normalizing_constant();
};


template< typename T >
template< typename InputIt >
stickygpm::matern<T>::matern( InputIt first, InputIt last ) :
  covariance_functor<T>(first, last)
{
  if ( std::distance(first, last) != 3 ) {
    throw std::domain_error(
      "matern functor requires 3 parameters");
  }
  _compute_normalizing_constant();
};




template< typename T >
void stickygpm::matern<T>::_compute_normalizing_constant() {
  _sqrt_2nu_rho = std::sqrt( 2 * nu() ) / rho();
  // _norm_c = variance() * std::pow( static_cast<T>(2), 1 - nu()) *
  //   std::pow( _sqrt_2nu_rho, nu() ) /
  //   std::tgamma( nu() );
  _norm_c = std::log( variance() ) +
    ( 1 - nu() ) * M_LN2 +
    nu() * std::log( _sqrt_2nu_rho ) -
    std::lgamma( nu() );
  //
  _norm_c = std::exp( _norm_c );
};



template< typename T >
T stickygpm::matern<T>::operator() ( const T val ) const {
  if ( val == 0 ) {
    return this->_par[0];
  }
  return _norm_c *
    std::pow( std::abs(val), nu() ) *
    boost::math::cyl_bessel_k( nu(), _sqrt_2nu_rho * std::abs(val) );
};



template< typename T >
T stickygpm::matern<T>::inverse( const T val ) const {
  T x = rho();
  T diff = std::numeric_limits<T>::infinity();
  int iter = 0;
  while ( std::abs(diff) > _tol  &&  iter < _max_it ) {
    diff = ( this->operator()(x) - val ) /
      stickygpm::gradient_xf( *this, x );
    x -= diff;
    iter++;
  }
  if ( iter >= _max_it && diff > _tol ) {
    std::cerr << "  ** Did not converge\n";
  }
  return x;
};


template< typename T >
T stickygpm::matern<T>::fwhm() const {
  return 2 * std::abs( inverse( variance() / 2 ) );
};



template< typename T >
std::vector<T>
stickygpm::matern<T>::gradient( const T val ) const {
  const T c = operator()( val );
  param_type theta = this->_par;
  matern<T> cov_tilde;
  std::vector<T> grad( 3 );
  grad[0] = c / variance();
  //
  theta[1] += _eps;
  cov_tilde.param( theta );
  grad[1] = (cov_tilde(val) - c) / _eps;
  //
  theta[1] -= _eps;
  theta[2] += _eps;
  cov_tilde.param( theta );
  grad[2] = (cov_tilde(val) - c) / _eps;
  //
  return grad;
};




template< typename T >
std::vector<T>
stickygpm::matern<T>::param_lower_bounds() const {
  std::vector<T> bound{ 0, 0, 0 };
  return bound;
};


template< typename T >
std::vector<T>
stickygpm::matern<T>::param_upper_bounds() const {
  std::vector<T> bound( 3, static_cast<T>(HUGE_VAL) );
  return bound;
};




template< typename T >
T stickygpm::matern<T>::variance() const {
  return this->_par[0];
};

template< typename T >
T stickygpm::matern<T>::nu() const {
  return this->_par[2];
};


template< typename T >
T stickygpm::matern<T>::rho() const {
  return this->_par[1];
};



template< typename T >
void stickygpm::matern<T>::param(
  const stickygpm::matern<T>::param_type& par
) {
  assert( par.size() == 3 && "Invalid parameter size" );
  assert( par[0] > 0 && "Invalid parameter (0)");
  assert( par[1] > 0 && "Invalid parameter (1)");
  assert( par[2] > 0 && "Invalid parameter (2)");
  this->_par = par;
  _compute_normalizing_constant();
};



template< typename T >
void stickygpm::matern<T>::variance( const T val ) {
  assert( val > 0 && "Invalid variance parameter");
  this->_par[0] = val;
  _compute_normalizing_constant();
};


template< typename T >
void stickygpm::matern<T>::nu( const T val ) {
  assert( val > 0 && "Invalid order parameter");
  this->_par[2] = val;
  _compute_normalizing_constant();
};


template< typename T >
void stickygpm::matern<T>::rho( const T val ) {
  assert( val > 0 && "Invalid inverse bandwidth parameter");
  this->_par[1] = val;
  _compute_normalizing_constant();
};




#endif  // _STICKYGPM_COVARIANCE_FUNCTORS_
