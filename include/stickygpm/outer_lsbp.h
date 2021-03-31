
#include <abseil/assignment_problem.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iomanip>
#include <iterator>
#include <random>
#include <limits>
#include <numeric>
// #include <omp.h>
#include <stdexcept>
#include <string>
#include <vector>


#include <iostream>

#include "stickygpm/constants.h"
#include "stickygpm/eigen_slicing.h"
#include "stickygpm/extra_distributions.h"
#include "stickygpm/logistic_weight_distribution.h"
#include "stickygpm/stickygpm_regression_data.h"
#include "stickygpm/truncated_logistic_distribution.h"
#include "stickygpm/truncated_normal_distribution.h"
#include "stickygpm/utilities.h"


#ifndef _STICKYGPM_OUTER_LSBP_
#define _STICKYGPM_OUTER_LSBP_





template< class InnerModelType >
class outer_lsbp {
public:
  typedef InnerModelType inner_model_type;
  typedef typename inner_model_type::scalar_type scalar_type;
  typedef typename
    Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>
    matrix_type;
  // typedef typename Eigen::BDCSVD<matrix_type> svd_type;
  typedef typename Eigen::Matrix<scalar_type, Eigen::Dynamic, 1>
    vector_type;

  outer_lsbp() { ; }
  outer_lsbp(
    const stickygpm::stickygpm_regression_data<scalar_type>& data,
    const int trunc = 20,                   // LSBP truncation
    const scalar_type sigma = SQRT_LN2_2,  // (Normal) prior scale for W
    const scalar_type mu0 = -Q_LOGIS_0_05  // (Normal) prior loc. for W
  );


  // --- Utility -----------------------------------------------------
  template< class Iter >
  void move_models( Iter first, Iter last );

  void update(
    const stickygpm::stickygpm_regression_data<scalar_type>& data,
    const vector_type& sigma_sq_inv,
    const double pr_use_likelihood = 0.95,
    const bool update_reference_labels = false
  );

  void initialize_clusters(
    const stickygpm::stickygpm_regression_data<scalar_type>& data
  );

  void sort_clusters();

  const InnerModelType& inner_model_ref( const int which ) const;


  // --- Output ------------------------------------------------------
  const matrix_type& logistic_coefficients() const;
  // matrix_type predict(const matrix_type& X) const;

  vector_type cluster_parameters( const int which ) const;
  vector_type cluster_sizes() const;
  vector_type realized_cluster_probability() const;
  vector_type residuals(
    const stickygpm::stickygpm_regression_data<scalar_type>& data,
    const int& subject_i
  ) const;

  double log_likelihood(
    const stickygpm::stickygpm_regression_data<scalar_type>& data,
    const vector_type& sigma_sq_inv
  ) const;
  
  double log_likelihood(
    const stickygpm::stickygpm_regression_data<scalar_type>& data,
    const vector_type& sigma_sq_inv,
    const int& subject_i
  ) const;

  double log_prior() const;

  int truncation() const;
  int cluster_label( const int subject_i ) const;  
  int occupied_clusters() const;

  template< typename OStream >
  void print_cluster_sizes( OStream& ost ) const;
  
  
private:

  bool _Initialized_;  // <- unused (yet)
  double _ref_log_posterior;
  double _log_prior;
  int _MaxClust_;

  // Cluster atoms / labels / indices
  std::vector<inner_model_type> _inner_models;
  std::vector<int> _cluster_labels;
  std::vector< std::vector<int> > _cluster_indices;
  //

  // Parameters related to updating the cluster labels
  std::vector<int> _reference_labels;
  std::vector<int> _clustering_remap;
  std::vector<double> _pr_clust_i;  // Pr(cluster assignment)
  //

  // Parameters related to updating of the logistic coefficients
  matrix_type _sigma2_inv;   // Prior scale of W  
  matrix_type _w_mu0;        // Prior means of W
  vector_type _eta;
  vector_type _lambda_inv;
  //
  
  vector_type _rgauss;        // Holds draw from Std. Gaussian
  
  matrix_type _PrecW;        // Precision of the _LoCoeffs_W_j
  matrix_type _LoCoeffs_W;   // Matrix of logistic model coefficients

  Eigen::MatrixXi _Clustering_cost;

  Eigen::LLT<matrix_type> _llt_PrecW;
  

  void _initialize_cluster_labels( const matrix_type& Y );

  void _update_cluster_labels(
    const stickygpm::stickygpm_regression_data<scalar_type>& data,
    const vector_type& sigma_sq_inv,
    const double pr_use_likelihood = 0.95,
    const bool update_reference_labels = false
  );

  void _update_logistic_coefficients( 
    const stickygpm::stickygpm_regression_data<scalar_type>& data
  );
  void _update_logistic_hyperparameters(
    const std::vector<Eigen::VectorXi>& random_effects_indices
  );

  
  void _draw_gaussian();

  void _reorder_clusters( const std::vector<int>& new_labels );
  void _reserve_cluster_indices( const int n );
  void _shrink_cluster_indices();
  
};






template< class InnerModelType >
outer_lsbp<InnerModelType>::outer_lsbp(
  const stickygpm::stickygpm_regression_data<
    typename outer_lsbp<InnerModelType>::scalar_type >& data,
  const int trunc,
  const typename outer_lsbp<InnerModelType>::scalar_type sigma,
  const typename outer_lsbp<InnerModelType>::scalar_type mu0
) {
  const int P = data.Z().cols();
  if ( trunc <= 0 ) {
    throw std::domain_error("LSBP truncation should be > 0");
  }
  _Initialized_ = false;
  _ref_log_posterior = -std::numeric_limits<double>::infinity();
  _log_prior = -std::numeric_limits<double>::infinity();
  _MaxClust_ = trunc - 1;
  _inner_models.reserve( trunc );
  _pr_clust_i.resize( trunc );


  _w_mu0 = matrix_type::Zero( P, trunc );
  _sigma2_inv = matrix_type::Constant( P, trunc, 1 / (sigma * sigma) );
  if ( data.lsbp_has_global_intercept() ) {
    for ( int j = 0; j < trunc; j++ ) {
      _w_mu0.coeffRef(0, j) = mu0 / (sigma * sigma);
      // _sigma2_inv.coeffRef(0, j) = 1e-4;  // don't penalize intercepts
    }
  }
  // _X = X;
  _LoCoeffs_W = matrix_type::Zero( P, trunc );
  std::normal_distribution<scalar_type> Gaussian( (scalar_type)0, sigma );
  for (int i = 0; i < _LoCoeffs_W.rows(); i++) {
    for (int j = 0; j < _LoCoeffs_W.cols(); j++) {
      _LoCoeffs_W.coeffRef(i, j) = Gaussian( stickygpm::rng() ) + mu0;
    }
  }
  
  _eta = vector_type::Zero( data.Z().rows() );
  _lambda_inv = vector_type::Ones( data.Z().rows() );
  logistic_weight_distribution<scalar_type> lwd(1);
  for (int i = 0; i < _lambda_inv.size(); i++) {
    _lambda_inv.coeffRef(i) = 1 / lwd( stickygpm::rng() );
  }

  _llt_PrecW = ( data.Z().adjoint() * data.Z() ).llt();

  _rgauss = vector_type::Zero( _LoCoeffs_W.rows() );
};







template< class InnerModelType >
template< class Iter >
void outer_lsbp<InnerModelType>::move_models(
  Iter first,
  Iter last
) {
  if ( (int)std::distance(first, last) != (_MaxClust_ + 1) ) {
    std::cerr
      << "WARNING: outer_lsbp: bad number of inner models moved"
      << std::endl;
  }
  std::move(first, last, std::back_inserter(_inner_models));
};


// template< class InnerModelType >
// void outer_lsbp<InnerModelType>::push_back_model(InnerModelType* mod) {
//   _inner_models.push_back(mod);
// };





template< class InnerModelType >
const typename outer_lsbp<InnerModelType>::matrix_type&
outer_lsbp<InnerModelType>::logistic_coefficients() const {
  return _LoCoeffs_W;
};






// template< class InnerModelType >
// typename outer_lsbp<InnerModelType>::matrix_type
// outer_lsbp<InnerModelType>::predict(
//   const typename outer_lsbp<InnerModelType>::matrix_type& Z
// ) const {
//   matrix_type Pred(_inner_models[0].xcor_matrix().cols(), Z.rows());  
//   Eigen::MatrixXd mPrk = ( X * _LoCoeffs_W ).template cast<double>();
//   std::vector<double> Pr_k_hat(mPrk.cols());
//   double remaining_stick;
//   int cluster;
//   for (int i = 0; i < X.rows(); i++) {
//     remaining_stick = 1;
//     for (int j = 0; j < mPrk.cols(); j++) {
//       mPrk.coeffRef(i, j) =
// 	extra_distributions::std_logistic_cdf(mPrk.coeffRef(i, j));
//       Pr_k_hat[j] = mPrk.coeffRef(i, j) * remaining_stick;
//       remaining_stick *= ( 1 - mPrk.coeffRef(i, j) );
//     }
//     std::discrete_distribution<int> Categorical(
//       Pr_k_hat.begin(), Pr_k_hat.end() );
//     cluster = Categorical(stickygpm::rng());
//     std::cout << "; " << cluster;
//     Pred.col(i) = _inner_models[cluster].mu();
//   }
//   return Pred;
// };






template< class InnerModelType >
typename outer_lsbp<InnerModelType>::vector_type
outer_lsbp<InnerModelType>::cluster_parameters( const int which )
  const {   
  if ( which < 0 || which > _MaxClust_ ) {
    throw std::domain_error("Argument 'which' out of scope");
  }
  return _inner_models[which].parameters();
};



template< class InnerModelType >
typename outer_lsbp<InnerModelType>::vector_type
outer_lsbp<InnerModelType>::cluster_sizes() const {
  vector_type n(_cluster_indices.size());
  for (int j = 0; j < _cluster_indices.size(); j++) {
    n.coeffRef(j) = (scalar_type)_cluster_indices[j].size();
  }
  return n;
};




template< class InnerModelType >
typename outer_lsbp<InnerModelType>::vector_type
outer_lsbp<InnerModelType>::realized_cluster_probability() const {
  vector_type prk(_cluster_indices.size());
  scalar_type total = 0;
  for (int j = 0; j < _cluster_indices.size(); j++) {
    prk.coeffRef(j) = _cluster_indices[j].size();
    total += prk.coeffRef(j);
  }
  return prk / total;
};






template< class InnerModelType >
typename outer_lsbp<InnerModelType>::vector_type
outer_lsbp<InnerModelType>::residuals(
  const stickygpm::stickygpm_regression_data<
    typename outer_lsbp<InnerModelType>::scalar_type >& data,
  const int& subject_i
) const {
#ifndef DNDEBUG
  if ( subject_i < 0 || subject_i >= _cluster_labels.size() ) {
    throw std::domain_error(
      "outer_lsbp::log_likelihood : bad subject index"
      );
  }
#endif
  // Cluster labels are always updated in data.Y()'s column order
  const int cluster = _cluster_labels[subject_i];
  return _inner_models[cluster].residuals( data, subject_i );
};








template< class InnerModelType >
double outer_lsbp<InnerModelType>::log_likelihood(
  const stickygpm::stickygpm_regression_data<
    typename outer_lsbp<InnerModelType>::scalar_type >& data,
  const typename outer_lsbp<InnerModelType>::vector_type& sigma_sq_inv
) const {
  double loglik = 0;
  int cluster;
  // Cluster labels are always updated in data.Y()'s column order
  for (int i = 0; i < data.n(); i++) {
    cluster = _cluster_labels[i];
    loglik += _inner_models[cluster].log_likelihood(
      data, sigma_sq_inv, i );
  }
  return loglik;
};






template< class InnerModelType >
double outer_lsbp<InnerModelType>::log_likelihood(
  const stickygpm::stickygpm_regression_data<
    typename outer_lsbp<InnerModelType>::scalar_type >& data,
  const typename outer_lsbp<InnerModelType>::vector_type& sigma_sq_inv,
  const int& subject_i
) const {
#ifndef DNDEBUG
  if ( subject_i < 0 || subject_i >= _cluster_labels.size() ) {
    throw std::domain_error(
      "outer_lsbp::log_likelihood : bad subject index"
      );
  }
#endif
  // Cluster labels are always updated in data.Y()'s column order
  const int cluster = _cluster_labels[subject_i];
  const double loglik = _inner_models[cluster].log_likelihood(
    data, sigma_sq_inv, subject_i );
  return loglik;
};





template< class InnerModelType >
double outer_lsbp<InnerModelType>::log_prior() const {
  return _log_prior;
};



template< class InnerModelType >
int outer_lsbp<InnerModelType>::truncation() const {
  return _MaxClust_ + 1;
};



template< class InnerModelType >
int outer_lsbp<InnerModelType>::cluster_label(
  const int subject_i
) const {
#ifndef DNDEBUG
  if ( subject_i < 0 || subject_i >= _cluster_labels.size() ) {
    throw std::domain_error(
      "outer_lsbp::cluster_label : bad subject index"
      );
  }
#endif
  return _cluster_labels[subject_i];
};






template< class InnerModelType >
int outer_lsbp<InnerModelType>::occupied_clusters() const {
  int n = 0;
  for (int j = 0; j < _cluster_indices.size(); j++) {
    if ( !_cluster_indices[j].empty() )
      n++;
  }
  return n;
};





template< class InnerModelType >
template< typename OStream >
void outer_lsbp<InnerModelType>::print_cluster_sizes(
  OStream& ost
) const {
  vector_type cs = cluster_sizes();
  for ( int i = 0; i < cs.size(); i++ ) {
    if ( cs.coeffRef(i) > 0 ) {
      ost << "{#" << i << " - "
	  << static_cast<int>(cs.coeffRef(i))
	  << "}  ";
    }
  }
};




template< class InnerModelType >
void outer_lsbp<InnerModelType>::update(
  const stickygpm::stickygpm_regression_data<
    typename outer_lsbp<InnerModelType>::scalar_type >& data,
  const typename outer_lsbp<InnerModelType>::vector_type& sigma_sq_inv,
  const double pr_use_likelihood,
  const bool update_reference_labels
) {
  const int n_lsbp_updates_ = 20;
  _log_prior = 0;
  // 1) Update inner models given cluster indices
  // double diff_sec;
  // auto start_t = std::chrono::high_resolution_clock::now();
  for ( int j = 0; j < _inner_models.size(); j++ ) {
    // std::cout << j << " ";
    if ( _cluster_indices[j].empty() ) {
      _inner_models[j].sample_from_prior();
    }
    else {
      _inner_models[j].update(
        data, sigma_sq_inv, _cluster_indices[j]
      );
    }
    _log_prior += _inner_models[j].log_prior();
  }
  // std::cout << std::endl;
  // auto stop_t = std::chrono::high_resolution_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::microseconds>
  //   (stop_t - start_t);
  // diff_sec = static_cast<double>(duration.count()) / 1e6;

  // std::cout << std::setprecision(4) << std::fixed
  // 	    << "\t.::| Inner - " << diff_sec << " (sec)  |  ";
  

  
  // 2) Update LSBP coefficients
  // start_t = std::chrono::high_resolution_clock::now();
  for ( int i = 0; i < n_lsbp_updates_; i++ ) {
    _update_logistic_coefficients( data );
  }
  _log_prior += -0.5 *
    ( _LoCoeffs_W.array() * _sigma2_inv.array().sqrt() ).matrix()
    .template cast<double>().colwise().squaredNorm().sum();
  _log_prior += 0.5 *
    _sigma2_inv.array().log().template cast<double>().sum();
  // std::cout << "W =\n" << _LoCoeffs_W
  // 	    << "\n\nmu0 =\n" << _w_mu0 << "\n\nsigma0 =\n"
  // 	    << _sigma2_inv.cwiseInverse() << std::endl;
  _update_logistic_hyperparameters(
    data.lsbp_random_effects_indices()
  );
  // stop_t = std::chrono::high_resolution_clock::now();
  // duration = std::chrono::duration_cast<std::chrono::microseconds>
  //   (stop_t - start_t);
  // diff_sec = static_cast<double>(duration.count()) / 1e6;

  // std::cout << "LSBP - " << diff_sec << " (sec)  |  ";
  
  
  // 3) Update cluster labels
  // start_t = std::chrono::high_resolution_clock::now();
  _update_cluster_labels(
    data,
    sigma_sq_inv,
    pr_use_likelihood,
    update_reference_labels
  );
  // ^^ Must come after _log_prior is set
  //
  // stop_t = std::chrono::high_resolution_clock::now();
  // duration = std::chrono::duration_cast<std::chrono::microseconds>
  //   (stop_t - start_t);
  // diff_sec = static_cast<double>(duration.count()) / 1e6;

  // std::cout << "Labels - " << diff_sec << " |::." << std::endl;
};








template< class InnerModelType >
void outer_lsbp<InnerModelType>::initialize_clusters(
  const stickygpm::stickygpm_regression_data<
    typename outer_lsbp<InnerModelType>::scalar_type >& data
) {
  _initialize_cluster_labels( data.Y() );
};






template< class InnerModelType >
void outer_lsbp<InnerModelType>::sort_clusters() {
  // Re-label reference clusters to place in descending-size order
  std::vector<int> ord( _cluster_indices.size() );
  std::vector<int> remap( ord.size() );
  std::vector<int> csize( _cluster_indices.size(), 0 );
  int L;
  // First count size of reference clusters
  for (int i = 0; i < _reference_labels.size(); i++) {
    csize[ _reference_labels[i] ]++;
  }
  //
  // Next order labels by reference cluster size
  std::iota( ord.begin(), ord.end(), 0 );
  std::sort( ord.begin(), ord.end(),
	     [&](int a, int b) -> bool {
	       return csize[a] > csize[b];
	     });
  // Find associated relabeling
  for (int k = 0; k < remap.size(); k++) {
    for (L = 0; ord[L] != k; L++) { ; }
    remap[k] = L;
  }
  // Execute relabelings:
  for ( int& rl : _reference_labels ) {
    rl = remap[ rl ];
  }
  //
  // std::cout << "For proposed sorting...\n";
  // for (int k = 0; k < ord.size(); k++) {
  //   std::cout << "  >> Label " << k << " -> " << remap[k]
  // 	      << "  ;  (n = " << csize[k]
  // 	      << ")\n";
  // }
  // std::cout << std::endl;
};







template< class InnerModelType >
const InnerModelType& outer_lsbp<InnerModelType>::inner_model_ref(
  const int which
) const {
  if ( which < 0 || which > _MaxClust_ ) {
    throw std::domain_error("Argument 'which' out of scope");
  }
  return _inner_models[which];
};




template< class InnerModelType >
void outer_lsbp<InnerModelType>::_initialize_cluster_labels(
  const typename outer_lsbp<InnerModelType>::matrix_type& Y
) {
  const int trunc = _MaxClust_ + 1;
  _cluster_labels.resize( Y.cols() );
  _reference_labels.resize( Y.cols() );
  _cluster_indices.resize( trunc );
  _reserve_cluster_indices( Y.cols() );
  _Clustering_cost = Eigen::MatrixXi::Zero( trunc, trunc );
  // _cluster_counts = std::vector<int>(trunc, 0);
  std::uniform_real_distribution _Uniform( 0, 1 );
  std::vector<vector_type> cluster_centers;
  cluster_centers.reserve( trunc );
  std::vector<double> cluster_probabilities;
  scalar_type mean_y, n_sigma2_y;
  int cluster, nclust, occupied_clusters = 0;
  const scalar_type E_INV = std::exp(-1);
  _cluster_labels[0] = 0;
  cluster_centers.push_back( Y.col(0).eval() );
  _cluster_indices[0].push_back(0);
  // _cluster_counts[0]++;
  occupied_clusters++;
  for (int i = 1; i < Y.cols(); i++) {
    cluster_probabilities.clear();
    cluster_probabilities.resize(
      std::min(occupied_clusters + 1, trunc));
    mean_y = Y.col(i).mean();
    n_sigma2_y = ( Y.col(i) -
		  vector_type::Constant(Y.rows(), mean_y) )
      .squaredNorm();
    for (int j = 0; j < occupied_clusters; j++) {
      cluster_probabilities[j] = std::exp( -0.5 *
        ( Y.col(i) - cluster_centers[j] ).squaredNorm() /
					   n_sigma2_y );
      // ^^ relative probabilities
    }
    if ( cluster_probabilities.size() < trunc ) {
      cluster_probabilities[cluster_probabilities.size() - 1] = E_INV;
    }
    std::discrete_distribution<int> Categorical(
      cluster_probabilities.begin(), cluster_probabilities.end());
    cluster = Categorical(stickygpm::rng());
    _cluster_labels[i] = cluster;
    _reference_labels[i] = cluster;
    _Clustering_cost.coeffRef(cluster, cluster) -= 1;
    _cluster_indices[cluster].push_back(i);
    // _cluster_counts[cluster]++;
    nclust = _cluster_indices[cluster].size();
    if ( i < (Y.cols() - 1) ) {
      // Only update cluster centers if needed for next iteration
      // if ( _cluster_counts[cluster] <= 1 ) {
      if ( nclust <= 1 ) {
	cluster_centers.push_back( Y.col(i).eval() );
	occupied_clusters++;
      }
      else {
	cluster_centers[cluster] =
	  ( cluster_centers[cluster] * (nclust - 1)
	    + Y.col(i) ) / nclust;
      }
    }
    // if ( i < (Y.cols() - 1) )
  }
  // for (int i = 1; i < Y.cols(); i++)  ...
  _shrink_cluster_indices();
  _Initialized_ = true;
};






template< class InnerModelType >
void outer_lsbp<InnerModelType>::_update_cluster_labels(
  const stickygpm::stickygpm_regression_data<
    typename outer_lsbp<InnerModelType>::scalar_type >& data,
  const typename outer_lsbp<InnerModelType>::vector_type& sigma_sq_inv,
  const double pr_use_likelihood,
  const bool update_reference_labels
) {
  // const double eps0 = 1e-12;
  const int N = data.n();
  Eigen::VectorXd vPrk(_LoCoeffs_W.cols());
  std::vector<double> likelihood(_LoCoeffs_W.cols());
  double remaining_stick, highest_loglik, sumlike;
  int cluster;
  long int max_cluster_size = 0;

  bool use_likelihood;
  std::uniform_real_distribution<> Uniform(0, 1);
  
  _Clustering_cost.setZero( _MaxClust_ + 1, _MaxClust_ + 1 );

  // Reserve space for cluster indices
  for (int j = 0; j < _cluster_indices.size(); j++) {
    if ( !_cluster_indices[j].empty() ) {
      max_cluster_size = std::max(
        max_cluster_size, (long)_cluster_indices[j].size() );
    }
  }
  _reserve_cluster_indices(
    std::min( (int)(1.2 * max_cluster_size), N ));
  // ^^ The first cluster should typically be the largest
  //


  
  for (int i = 0; i < N; i++) {
    //
    // std::cout << "\t" << i << ": Prior = (";
    //
    vPrk = ( data.Z().row(i) * _LoCoeffs_W ).template cast<double>();
    remaining_stick = 1;
    highest_loglik = 0;
    sumlike = 0;

    use_likelihood = (Uniform(stickygpm::rng()) < pr_use_likelihood);
    
    for (int j = 0; j < vPrk.size(); j++) {
      vPrk.coeffRef(j) = isnan( vPrk.coeffRef(j) ) ? 0 :
	extra_distributions::std_logistic_cdf( vPrk.coeffRef(j) );
      _pr_clust_i[j] = vPrk.coeffRef(j) * remaining_stick;
      remaining_stick *= ( 1 - vPrk.coeffRef(j) );

      
      if ( use_likelihood ) {
	
	// first store the log likelihood
	likelihood[j] = _inner_models[j].log_likelihood(
          data, sigma_sq_inv, i );
	
	if ( isnan(likelihood[j]) ) {
	  likelihood[j] = -std::numeric_limits<double>::infinity();
	}
	// mean_loglik += likelihood[j];
	highest_loglik = (j == 0) ?
	  likelihood[j] : std::max(highest_loglik, likelihood[j]);
	
      }  // if (use_likelihood)
      
      //
      // std::cout << _pr_clust_i[j] << ", ";
      //
      
    }  // for (int j = 0; j < vPrk.size(); j++)


    if ( use_likelihood ) {
      //
      // std::cout << "\b\b)   :   Likelihood = (";
      //
      for (int j = 0; j < likelihood.size(); j++) {
	likelihood[j] = std::exp(likelihood[j] - highest_loglik);
	//
	// std::cout << likelihood[j] << ", ";
	//
	sumlike += likelihood[j];
	// std::cout << likelihood[j] << "  ";
      }
      // std::cout << std::endl;
      for (int j = 0; j < _pr_clust_i.size(); j++) {
	_pr_clust_i[j] *= likelihood[j] / sumlike;
	//
	// std::cout << (likelihood[j] / sumlike) << ", ";
	//
      }
      //
      // std::cout << "\b\b) " << std::endl;
      //

    }  // if (use_likelihood)

    
    std::discrete_distribution<int> Categorical(
      _pr_clust_i.begin(), _pr_clust_i.end() );
    cluster = Categorical( stickygpm::rng() );
    // Update reference/active cluster overlap matrix
    _Clustering_cost
      .coeffRef(_reference_labels[i], cluster) -= 1;
    //
    _cluster_labels[i] = cluster;
    _cluster_indices[cluster].push_back(i);
    // _cluster_counts[cluster]++;
  }
  // end - for (int i = 0; i < N; i++)  ...
  _shrink_cluster_indices();
  //
  if ( update_reference_labels ) {
    // repurpose sumlike into this iteration's log likelihood
    sumlike = log_likelihood( data, sigma_sq_inv ) + _log_prior;
    if ( sumlike > _ref_log_posterior ) {
      _ref_log_posterior = sumlike;
      _reference_labels.assign(
        _cluster_labels.begin(), _cluster_labels.end());
    }
  }
  else {
    // Only compute clustering re-alignment if Reference
    // Labels have already been fixed
    //
    // Relabel clusters based on reference
    abseil::min_cost_assignment::solve(
      _Clustering_cost,
      _clustering_remap  // <- modified
    );
    //
    _reorder_clusters( _clustering_remap );
  }
  // for (int i = 0; i < _cluster_labels.size(); i++) {
  //   if ( !update_reference_labels ) {
  //     _cluster_labels[i] =
  //       _clustering_remap[_cluster_labels[i]];
  //   }
  //   _cluster_indices[_cluster_labels[i]].push_back(i);
  // }
  //
  //
  // std::cout << std::endl;
  //
};








// template< class InnerModelType >
// void outer_lsbp<InnerModelType>::_update_decompositions(
//   const typename outer_lsbp<InnerModelType>::matrix_type& X
// ) {
//   Eigen::VectorXi all_cols_X_ =
//     Eigen::VectorXi::LinSpaced(X.cols(), 0, X.cols() - 1);
//   for (int k = 0; k < _cluster_indices.size(); k++) {
//     if ( !_cluster_indices[k].empty() ) {
//       Eigen::VectorXi k_ind = Eigen::Map<Eigen::VectorXi>(
//         _cluster_indices[k].data(), _cluster_indices[k].size());
//       _clustered_X_svd[k] = svd_type(
//         stickygpm::nullary_index(X, k_ind, all_cols_X_),
// 	Eigen::DecompositionOptions::ComputeThinU |
// 	Eigen::DecompositionOptions::ComputeThinV
//       );
//     }
//   }
// };







template< class InnerModelType >
void outer_lsbp<InnerModelType>::_update_logistic_coefficients(
  const stickygpm::stickygpm_regression_data<
    typename outer_lsbp<InnerModelType>::scalar_type >& data
) {
  const int N = data.Z().rows();
  int cluster, n_in_clust;
  scalar_type residual, residual2, upper, lower, mu;
  const scalar_type eps0 = 1e-4;
  
  for (int j = 0; j < _LoCoeffs_W.cols(); j++) {  // Outer loop over clusters

    n_in_clust = _cluster_indices[j].size();
    // std::cout << n_in_clust << "  ";
    if ( n_in_clust > 0  &&  n_in_clust < N ) {
      // Update with likelihood info
      
      for (int i = 0; i < _cluster_labels.size(); i++) {
	cluster = _cluster_labels[i];
	mu = data.Z().row(i) * _LoCoeffs_W.col(j);
	mu = std::isinf( mu ) ? 100 : mu;
	if ( cluster == j ) {
	  upper = 1e4;
	  lower = 0;
	}
	else {
	  // else if (cluster > j) {
	  upper = 0;
	  lower = -1e4;
	}
	// else {
	//   upper = 1e4;
	//   lower = -1e4;
	// }
	truncated_logistic_distribution<scalar_type> TLogis(
          mu, 1, lower, upper);
	_eta.coeffRef(i) = TLogis( stickygpm::rng() );
	residual = _eta.coeffRef(i) - mu;
	residual2 = isnan(residual) ? eps0 : (residual * residual);
	logistic_weight_distribution<scalar_type> LWD( residual2 );
	_lambda_inv.coeffRef(i) = 1 / LWD( stickygpm::rng() );
      }
      // for (int i = 0; ...
      // (loop over participants)
    
      _draw_gaussian();
      _PrecW = data.Z().transpose() * _lambda_inv.asDiagonal() * data.Z();
      _PrecW +=  _sigma2_inv.col(j).asDiagonal();
      //  ^^ Eigen doesn't like doing this addition with the previous step
      //     since Z is const qualified
      _llt_PrecW = _PrecW.llt();
      // _llt_PrecW = ( (X.transpose() * _lambda_inv.asDiagonal() * X).eval() +
      //   _sigma2_inv.asDiagonal() ).llt();
      _LoCoeffs_W.col(j) = _llt_PrecW.solve(
        _sigma2_inv.col(j).asDiagonal() * _w_mu0.col(j) +
	data.Z().transpose() * _lambda_inv.asDiagonal() * _eta +
	_llt_PrecW.matrixL() * _rgauss
      );

    }
    else {  // Either no or all observations are assigned to cluster
      if ( n_in_clust == 0) {
      	lower = -1e3;
      	upper = (scalar_type)extra_distributions
      	  ::std_logistic_quantile( (double)1 / (N + 1) );
      }
      else {
      	lower = (scalar_type)extra_distributions
      	  ::std_logistic_quantile( (double)N / (N + 1) );
      	upper = 1e3;
      }
        // _w_mu0.coeffRef(0),
      	// 1 / _sigma2_inv.coeffRef(0),
      _LoCoeffs_W.col(j) = vector_type::Zero( _LoCoeffs_W.rows() );
      if ( data.lsbp_has_global_intercept() ) {
	truncated_normal_distribution<scalar_type> TNorm(
          (scalar_type)0,  // _w_mu0.coeffRef(0)
	  (scalar_type)1,
	  // _w_mu0.coeffRef(0, j),
	  // 1 / _sigma2_inv.coeffRef(0, j),
	  lower, upper
        );
	_LoCoeffs_W.coeffRef(0, j) = TNorm( stickygpm::rng() );
      }
      //
      // ---
      //
      //  !!!  (There's no global intercept in current example!)
      //
      // _draw_gaussian();
      // _LoCoeffs_W.col(j) = _w_mu0.col(j) +
      // 	_sigma2_inv.col(j).array().pow(-0.5).matrix().asDiagonal() *
      // 	_rgauss;
      // for (int k = 0; k < _LoCoeffs_W.rows(); k++)
      // 	_LoCoeffs_W.coeffRef(k, j) += 2 * (scalar_type)_Q_LOGIS_0_05;
      //
      // "Approximate" prior sample -
      // If no observations have been assigned to the cluster,
      // then the latent 0/1 data is all 0. Adding 4 * qlogis(0.05)
      // to the intercept with samples from the prior in this column
      // approximates a low probability cluster
      // if ( data.lsbp_has_global_intercept() ) {
      // 	_LoCoeffs_W.coeffRef(0, j) += 4 * (scalar_type)_Q_LOGIS_0_05;
      // }
      
    }  // if/else ( n_in_clust > 0 )
    
  }
  // std::cout << std::endl;
  // for (int j = 0; ...
  // (loop over clusters)
};




template< class InnerModelType >
void outer_lsbp<InnerModelType>::_update_logistic_hyperparameters(
    const std::vector<Eigen::VectorXi>& random_effects_indices
) {
  // typedef std::vector<Eigen::VectorXi>::const_iterator re_iter;
  if ( !random_effects_indices.empty() ) {
    std::normal_distribution<scalar_type> Gaussian(0, 1);
    scalar_type w, sum_w, sum_w2;
    scalar_type mu, tau, tau_rate;
    int n;
    
    for ( const Eigen::VectorXi& indices : random_effects_indices ) {
      n = indices.size();
      for ( int j = 0; j < _LoCoeffs_W.cols(); j++ ) {

	if ( !_cluster_indices[j].empty() ) {
	  tau = _sigma2_inv.coeffRef( indices.coeffRef(0), j );
	  sum_w = 0;
	  sum_w2 = 0;
	  for ( int i = 0; i < n; i++ ) {
	    w = _LoCoeffs_W.coeffRef( indices.coeffRef(i), j );
	    sum_w += w;
	    sum_w2 += w * w;
	  }
	  mu = sum_w / n + std::sqrt( 1 / (n * tau) ) *
	    Gaussian( stickygpm::rng() );
	  tau_rate = 0.5 * (sum_w2 - 2 * mu * sum_w + n * mu * mu) + 1;

	  std::gamma_distribution<scalar_type> Gamma(
            0.5 * n + 1, 1 / tau_rate
          );
	  tau = Gamma( stickygpm::rng() );

	  for ( int i = 0; i < n; i++ ) {
	    _w_mu0.coeffRef( indices.coeffRef(i), j ) = mu;
	    _sigma2_inv.coeffRef( indices.coeffRef(i), j ) = tau;
	  }
	}
	// if ( !_cluster_indices[j].empty() )
	
      }
      // for ( int j = 0; j < _LoCoeffs_W.cols(); j++ )
    }
    // for ( const Eigen::VectorXi& indices : random_effects_indices )
    
  }
  // if ( !random_effects_indices.empty() )
};





template< class InnerModelType >
void outer_lsbp<InnerModelType>::_draw_gaussian() {
  std::normal_distribution<scalar_type> Gaussian(0, 1);
  for (int i = 0; i < _rgauss.size(); i++) {
    _rgauss[i] = Gaussian( stickygpm::rng() );
  }
};


// template< class InnerModelType >
// void outer_lsbp<InnerModelType>::_stickify() {
//   double remaining_stick = 1;
//   for ( std::vector<double>::iterator it = _pr_clust_i.begin();
// 	it != _pr_clust_i.end(); ++it ) {
//     (*it) *= remaining_stick;
//     remaining_stick *= ( 1 - (*it) );
//   }
// };



template< class InnerModelType >
void outer_lsbp<InnerModelType>::_reorder_clusters(
  const std::vector<int>& new_labels
) {
  std::vector<int> ord(new_labels.cbegin(), new_labels.cend());
  // ^^ Modified in keeping track of remaining indices to be ordered
  vector_type wtemp(_LoCoeffs_W.rows());
  // Reassign cluster labels
  for ( int& cl : _cluster_labels ) {
    cl = new_labels[ cl ];
  }
  //
  // Reorder cluster:
  //   atoms / indices / logistic coefficients
  for (int k = 0; k < new_labels.size(); k++) {
    // Swap inner model pointers
    std::iter_swap(
      _inner_models.begin() + k,
      _inner_models.begin() + ord[k]
    );
    // Swap pointers to cluster indices
    std::iter_swap(
      _cluster_indices.begin() + k,
      _cluster_indices.begin() + ord[k]
    );
    // Swap logistic coefficients manually
    wtemp = _LoCoeffs_W.col(k);
    _LoCoeffs_W.col(k) = _LoCoeffs_W.col(ord[k]);
    _LoCoeffs_W.col(ord[k]) = wtemp;
    //
    // Update 'ord'
    for (int h = k; h < ord.size(); h++) {
      if ( ord[h] == k ) {
	ord[h] = ord[k];
	break;
      }
    }
  }
  //
};




template< class InnerModelType >
void outer_lsbp<InnerModelType>::_reserve_cluster_indices(
  const int n
) {
  typedef std::vector< std::vector<int> >::iterator vit;
  for (vit it = _cluster_indices.begin();
       it != _cluster_indices.end(); ++it) {
    it->clear();
    it->reserve(n);
  }
};


template< class InnerModelType >
void outer_lsbp<InnerModelType>::_shrink_cluster_indices() {
  typedef std::vector< std::vector<int> >::iterator vit;
  for (vit it = _cluster_indices.begin();
       it != _cluster_indices.end(); ++it) {
    it->shrink_to_fit();
  }
};




#endif  // _STICKYGPM_OUTER_LSBP_

