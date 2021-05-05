
#include <abseil/assignment_problem.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>
#include <limits>
#include <numeric>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <vector>


#include "stickygpm/eigen_slicing.h"
#include "stickygpm/extra_distributions.h"
#include "stickygpm/logistic_weight_distribution.h"
#include "stickygpm/stickygpm_regression_data.h"
#include "stickygpm/truncated_logistic_distribution.h"
#include "stickygpm/truncated_normal_distribution.h"
#include "stickygpm/utilities.h"


#ifndef _STICKYGPM_OUTER_RLSBP_
#define _STICKYGPM_OUTER_RLSBP_




template< class InnerModelType >
class outer_rlsbp {
public:
  typedef InnerModelType inner_model_type;
  typedef typename inner_model_type::scalar_type scalar_type;
  typedef typename
    Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>
    matrix_type;
  // typedef typename
  //   Eigen::SparseMatrix<scalar_type, Eigen::RowMajor>
  //   sparse_matrix_type;
  // typedef typename Eigen::BDCSVD<matrix_type> svd_type;
  typedef typename Eigen::Matrix<scalar_type, Eigen::Dynamic, 1>
    vector_type;
  typedef typename inner_model_type::param_type param_type;

  outer_rlsbp() { ; }
  outer_rlsbp(
    const stickygpm::stickygpm_regression_data<scalar_type>& data,
    const int trunc = 20,         // LSBP truncation
    const scalar_type sigma = 1,  // (Normal) prior scale for W
    const scalar_type mu0 = 0,    // (Normal) prior loc. for W
    const double tau = 0          // Repulsion parameter
  );


  // --- Utility -----------------------------------------------------
  template< class Iter >
  void move_models( Iter first, Iter last );

  double update(
    const stickygpm::stickygpm_regression_data<scalar_type>& data,
    const vector_type& sigma_sq_inv,
    const double pr_use_likelihood = 0.95,
    const bool update_reference_labels = false,
    const bool align_labels_to_reference = false
  );

  void initialize_clusters(
    const stickygpm::stickygpm_regression_data<scalar_type>& data,
    const int k = 1
  );

  void sort_clusters();

  const InnerModelType& inner_model_ref( const int which ) const;


  // --- Output ------------------------------------------------------
  const matrix_type& logistic_coefficients() const;
  // matrix_type predict(const matrix_type& X) const;

  param_type cluster_parameter( const int which ) const;
  
  vector_type cluster_sizes() const;
  vector_type realized_cluster_probability() const;
  vector_type residuals(
    const stickygpm::stickygpm_regression_data<scalar_type>& data,
    const int& subject_i
  ) const;

  
  double log_kernel(
    const stickygpm::stickygpm_regression_data<scalar_type>& data,
    const vector_type& sigma_sq_inv,
    const int& subject_i
  ) const;
  
  double log_likelihood() const;
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
  double repulsion_parameter() const;
  double min_cluster_distance() const;

  int truncation() const;
  int cluster_label( const int subject_i ) const;  
  int occupied_clusters() const;
  int modal_cluster() const;

  void print_cluster_sizes( std::ostream& ost ) const;
  void print_reference_sizes( std::ostream& ost ) const;

  // void rejection_rate( const double rate );
  void repulsion_parameter( const double tau );
  
  
private:

  bool _Initialized_;  // <- unused (yet)
  bool _has_intercept;
  double _ref_log_posterior;
  double _log_prior;
  double _log_likelihood;
  double _tau_;        // Repulsion parameter, > 0
  double _min_cluster_dist_;
  // double _log_tau_bar_;
  // double _m_;
  // double _mh_target;
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
  matrix_type _loc_shrink;
  vector_type _glb_shrink;
  vector_type _eta;
  vector_type _lambda_inv;
  //
  
  vector_type _rgauss;        // Holds draw from Std. Gaussian
  
  matrix_type _PrecW;        // Precision of the _LoCoeffs_W_j
  matrix_type _LoCoeffs_W;   // Matrix of logistic model coefficients

  Eigen::MatrixXi _Clustering_cost;

  Eigen::LLT<matrix_type> _llt_PrecW;
  

  void _initialize_cluster_labels(
    const matrix_type& Y,
    const int k
  );
  void _initialize_single_cluster_label( const matrix_type& Y );

  void _update_cluster_labels(
    const stickygpm::stickygpm_regression_data<scalar_type>& data,
    const vector_type& sigma_sq_inv,
    const double pr_use_likelihood = 0.95,
    const bool update_reference_labels = false,
    const bool align_to_reference = false
  );

  void _update_logistic_coefficients( 
    const stickygpm::stickygpm_regression_data<scalar_type>& data
  );
  void _update_logistic_hyperparameters(
    const std::vector<Eigen::VectorXi>& random_effects_indices,
    const std::vector<Eigen::VectorXi>& fixed_effects_indices
  );

  
  void _draw_gaussian();

  void _reorder_clusters( const std::vector<int>& new_labels );
  void _reserve_cluster_indices( const int n );
  void _shrink_cluster_indices();
  
};






template< class InnerModelType >
outer_rlsbp<InnerModelType>::outer_rlsbp(
  const stickygpm::stickygpm_regression_data<
    typename outer_rlsbp<InnerModelType>::scalar_type >& data,
  const int trunc,
  const typename outer_rlsbp<InnerModelType>::scalar_type sigma,
  const typename outer_rlsbp<InnerModelType>::scalar_type mu0,
  const double tau
) {
  const int P = data.Z().cols();
  if ( trunc <= 0 ) {
    throw std::domain_error("LSBP truncation should be > 0");
  }
  _Initialized_ = false;
  _has_intercept = data.lsbp_has_global_intercept();
  
  _ref_log_posterior = -std::numeric_limits<double>::infinity();
  _log_prior = -std::numeric_limits<double>::infinity();
  _log_likelihood = -std::numeric_limits<double>::infinity();
  _MaxClust_ = trunc - 1;
  _inner_models.reserve( trunc );
  _pr_clust_i.resize( trunc );

  //
  _min_cluster_dist_ = 0;
  _tau_ = tau;
  // _m_ = 10;
  // _mh_target = mhtarget;
  //


  _w_mu0 = matrix_type::Zero( P, trunc );
  _sigma2_inv = matrix_type::Constant( P, trunc, 1 / (sigma * sigma) );
  if ( _has_intercept ) {
    for ( int j = 0; j < trunc; j++ ) {
      // _w_mu0.coeffRef(0, j) = mu0 / (sigma * sigma);
      // _sigma2_inv.coeffRef(0, j) = 1e-4;  // don't penalize intercepts
      _sigma2_inv.coeffRef(0, j) = 10 * sigma;
      // ^^ Milder penalty on intercepts
      _w_mu0.coeffRef(0, j) = mu0 * _sigma2_inv.coeffRef(0, j);
    }
  }
  //
  _loc_shrink = _sigma2_inv;
  _glb_shrink = vector_type::Ones( _sigma2_inv.cols() );
  //
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

  _PrecW = data.Z().adjoint() * data.Z();
  _llt_PrecW = _PrecW.llt();

  _rgauss = vector_type::Zero( _LoCoeffs_W.rows() );
};







template< class InnerModelType >
template< class Iter >
void outer_rlsbp<InnerModelType>::move_models(
  Iter first,
  Iter last
) {
  if ( (int)std::distance(first, last) != (_MaxClust_ + 1) ) {
    std::cerr
      << "WARNING: outer_rlsbp: bad number of inner models moved"
      << std::endl;
  }
  std::move(first, last, std::back_inserter(_inner_models));
};


// template< class InnerModelType >
// void outer_rlsbp<InnerModelType>::push_back_model(InnerModelType* mod) {
//   _inner_models.push_back(mod);
// };





template< class InnerModelType >
const typename outer_rlsbp<InnerModelType>::matrix_type&
outer_rlsbp<InnerModelType>::logistic_coefficients() const {
  return _LoCoeffs_W;
};






// template< class InnerModelType >
// typename outer_rlsbp<InnerModelType>::matrix_type
// outer_rlsbp<InnerModelType>::predict(
//   const typename outer_rlsbp<InnerModelType>::matrix_type& Z
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
typename outer_rlsbp<InnerModelType>::param_type
outer_rlsbp<InnerModelType>::cluster_parameter( const int which )
  const {   
  if ( which < 0 || which > _MaxClust_ ) {
    throw std::domain_error("Argument 'which' out of scope");
  }
  return _inner_models[ which ].parameter();
};



template< class InnerModelType >
typename outer_rlsbp<InnerModelType>::vector_type
outer_rlsbp<InnerModelType>::cluster_sizes() const {
  vector_type n(_cluster_indices.size());
  for (int j = 0; j < (int)_cluster_indices.size(); j++) {
    n.coeffRef(j) = (scalar_type)_cluster_indices[ j ].size();
  }
  return n;
};




template< class InnerModelType >
typename outer_rlsbp<InnerModelType>::vector_type
outer_rlsbp<InnerModelType>::realized_cluster_probability() const {
  vector_type prk(_cluster_indices.size());
  scalar_type total = 0;
  for (int j = 0; j < (int)_cluster_indices.size(); j++) {
    prk.coeffRef(j) = _cluster_indices[j].size();
    total += prk.coeffRef(j);
  }
  return prk / total;
};






template< class InnerModelType >
typename outer_rlsbp<InnerModelType>::vector_type
outer_rlsbp<InnerModelType>::residuals(
  const stickygpm::stickygpm_regression_data<
    typename outer_rlsbp<InnerModelType>::scalar_type >& data,
  const int& subject_i
) const {
#ifndef DNDEBUG
  if ( subject_i < 0 || subject_i >= (int)_cluster_labels.size() ) {
    throw std::domain_error(
      "outer_rlsbp::log_likelihood : bad subject index"
      );
  }
#endif
  // Cluster labels are always updated in data.Y()'s column order
  const int cluster = _cluster_labels[subject_i];
  return _inner_models[cluster].residuals( data, subject_i );
};








template< class InnerModelType >
double outer_rlsbp<InnerModelType>::log_kernel(
  const stickygpm::stickygpm_regression_data<
    typename outer_rlsbp<InnerModelType>::scalar_type >& data,
  const typename outer_rlsbp<InnerModelType>::vector_type& sigma_sq_inv,
  const int& subject_i
) const {
#ifndef DNDEBUG
  if ( subject_i < 0 || subject_i >= (int)_cluster_labels.size() ) {
    throw std::domain_error(
      "outer_rlsbp::log_likelihood : bad subject index"
      );
  }
#endif
  // Cluster labels are always updated in data.Y()'s column order
  const int cluster = _cluster_labels[subject_i];
  const double lk = _inner_models[cluster].log_kernel(
    data, sigma_sq_inv, subject_i );
  return lk;
};





template< class InnerModelType >
double outer_rlsbp<InnerModelType>::log_likelihood() const {
  return _log_likelihood;
};


template< class InnerModelType >
double outer_rlsbp<InnerModelType>::log_likelihood(
  const stickygpm::stickygpm_regression_data<
    typename outer_rlsbp<InnerModelType>::scalar_type >& data,
  const typename outer_rlsbp<InnerModelType>::vector_type& sigma_sq_inv
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
double outer_rlsbp<InnerModelType>::log_likelihood(
  const stickygpm::stickygpm_regression_data<
    typename outer_rlsbp<InnerModelType>::scalar_type >& data,
  const typename outer_rlsbp<InnerModelType>::vector_type& sigma_sq_inv,
  const int& subject_i
) const {
#ifndef DNDEBUG
  if ( subject_i < 0 || subject_i >= (int)_cluster_labels.size() ) {
    throw std::domain_error(
      "outer_rlsbp::log_likelihood : bad subject index"
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
double outer_rlsbp<InnerModelType>::log_prior() const {
  return _log_prior;
};



template< class InnerModelType >
double outer_rlsbp<InnerModelType>::repulsion_parameter() const {
  return _tau_;
};


template< class InnerModelType >
double outer_rlsbp<InnerModelType>::min_cluster_distance() const {
  return _min_cluster_dist_;
};


template< class InnerModelType >
int outer_rlsbp<InnerModelType>::truncation() const {
  return _MaxClust_ + 1;
};



template< class InnerModelType >
int outer_rlsbp<InnerModelType>::cluster_label(
  const int subject_i
) const {
#ifndef DNDEBUG
  if ( subject_i < 0 || subject_i >= (int)_cluster_labels.size() ) {
    throw std::domain_error(
      "outer_rlsbp::cluster_label : bad subject index"
      );
  }
#endif
  return _cluster_labels[subject_i];
};






template< class InnerModelType >
int outer_rlsbp<InnerModelType>::occupied_clusters() const {
  int n = 0;
  for (int j = 0; j < (int)_cluster_indices.size(); j++) {
    if ( !_cluster_indices[j].empty() )
      n++;
  }
  return n;
};




template< class InnerModelType >
int outer_rlsbp<InnerModelType>::modal_cluster() const {
  int which = 0;
  int nmax = _cluster_indices[0].size();
  for (int j = 1; j < (int)_cluster_indices.size(); j++) {
    if ( (int)_cluster_indices[j].size() > nmax ) {
      which = j;
      nmax = (int)_cluster_indices[j].size();
    }
  }
  return which;
};





template< class InnerModelType >
void outer_rlsbp<InnerModelType>::print_cluster_sizes(
  std::ostream& ost
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
void outer_rlsbp<InnerModelType>::print_reference_sizes(
  std::ostream& ost
) const {
  std::vector<int> rcs( _cluster_indices.size(), 0 );
  for ( int rl : _reference_labels ) {
    rcs[rl]++;
  }
  ost << "\tReference Labels:  ";
  for ( int i = 0; i < (int)rcs.size(); i++ ) {
    if ( rcs[i] > 0 ) {
      ost << "{#" << i << " - " << rcs[i] << "}  ";
    }
  }
  ost << "\n\tReference log-Posterior:  " << _ref_log_posterior
      << "\n";
};




template< class InnerModelType >
void outer_rlsbp<InnerModelType>::repulsion_parameter(
  const double tau
) {
  if ( tau < 0 ) {
    std::cerr << "outer_rlsbp: repulsion parameter not set: out of scope"
              << std::endl;
  }
  else {
    _tau_ = tau;
  }
};





template< class InnerModelType >
double outer_rlsbp<InnerModelType>::update(
  const stickygpm::stickygpm_regression_data<
    typename outer_rlsbp<InnerModelType>::scalar_type >& data,
  const typename outer_rlsbp<InnerModelType>::vector_type& sigma_sq_inv,
  const double pr_use_likelihood,
  const bool update_reference_labels,
  const bool align_labels_to_reference
) {
  const int n_lsbp_updates_ = 20;
  const double eps0 = 1e-12;
  _log_prior = 0;
  //  ***
  double ratio, A;
  double prop_dist, prop_min_dist = HUGE_VAL;
  bool accept;
  std::uniform_real_distribution<double> Unif(0, 1);
  //  ***
  // 1) Update inner models given cluster indices
  // double diff_sec;
  // auto start_t = std::chrono::high_resolution_clock::now();
  //
  for ( int j = 0; j < (int)_inner_models.size(); j++ ) {
    // std::cout << j << " ";
    if ( _cluster_indices[j].empty() ) {
      _inner_models[j].sample_from_prior();
    }
    else {
      _inner_models[j].update(
        data, sigma_sq_inv, _cluster_indices[j]
      );
    }
    //  ***
    if ( j > 0 ) {
      for ( int k = 0; k < j; k++ ) {
	prop_dist = _inner_models[k].proposal_distance(
          _inner_models[j].proposed_parameter()
        );
	prop_min_dist = ( prop_dist < prop_min_dist ) ? prop_dist : prop_min_dist;
      }
    }
    //  ***
  }
  //  ***
  // A = 1 / ( prop_min_dist * prop_min_dist + eps0 ) +
  //   -1 / ( _min_cluster_dist_ * _min_cluster_dist_ + eps0 );
  A = 1 / ( prop_min_dist * prop_min_dist + eps0 );
  ratio = std::exp( -_tau_ * A );
  ratio = isnan( ratio ) ? 0 : ratio;
  ratio = ( ratio > 1 ) ? 1 : ratio;
  accept = Unif( stickygpm::rng() ) < ratio;
  for ( int j = 0; j < (int)_inner_models.size(); j++ ) {
    if ( accept ) {
      _inner_models[j].accept_proposal();
    }
    _log_prior += _inner_models[j].log_prior();
  }
  if ( accept ) {
    _min_cluster_dist_ = prop_min_dist;  // For next iteration
  }
  // if ( update_repulsion ) {
  //   // bigger tau <==> lower MH rate
  //   _tau_ = std::abs( std::log( _mh_target ) / (A + eps0) );
  //   // _tau_ = ( _tau_ < 1 ) ? 1 : _tau_;
  //   _log_tau_bar_ = (1 - std::pow(_m_, -0.75)) * _log_tau_bar_ +
  //     std::pow(_m_, -0.75) * std::log( _tau_ );
  //   //
  //   _tau_ = std::exp( _log_tau_bar_ );
  //   _m_++;
  // }
  // std::cout << "(tau = " << _tau_ << ")\n";
  // std::cout << "(half step _log_prior = " << _log_prior << ")\n";
  //  ***
  // std::cout << std::endl;
  // auto stop_t = std::chrono::high_resolution_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::microseconds>
  //   (stop_t - start_t);
  // diff_sec = static_cast<double>(duration.count()) / 1e6;

  // std::cout << std::setprecision(4) << std::fixed
  // 	    << "\t.::| Inner - " << diff_sec << " (sec)  |  ";
  

  
  // 2) Update LSBP coefficients
  // start_t = std::chrono::high_resolution_clock::now();
  //
  for ( int i = 0; i < n_lsbp_updates_; i++ ) {
    _update_logistic_coefficients( data );
  }
  //
  // std::cout << "\n(" << _log_prior << ")\n";
  //
  _log_prior += -0.5 *
    ( _LoCoeffs_W.array() * _sigma2_inv.array().sqrt() ).matrix()
    .template cast<double>().colwise().squaredNorm().sum();
  //
  // std::cout << _LoCoeffs_W << "\n";
  // std::cout << _sigma2_inv << "\n";
  // std::cout << "\n(" << _log_prior << ")\n";
  //
  _log_prior += 0.5 *
    _sigma2_inv.array().template cast<double>().log().sum();
  // std::cout << "W =\n" << _LoCoeffs_W
  // 	    << "\n\nmu0 =\n" << _w_mu0 << "\n\nsigma0 =\n"
  // 	    << _sigma2_inv.cwiseInverse() << std::endl;
  _update_logistic_hyperparameters(
    data.lsbp_random_effects_indices(),
    data.lsbp_fixed_effects_indices()
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
    update_reference_labels,
    align_labels_to_reference
  );
  // ^^ Must come after _log_prior is set
  //
  // stop_t = std::chrono::high_resolution_clock::now();
  // duration = std::chrono::duration_cast<std::chrono::microseconds>
  //   (stop_t - start_t);
  // diff_sec = static_cast<double>(duration.count()) / 1e6;

  // std::cout << "Labels - " << diff_sec << " |::." << std::endl;

  
  // std::cout << "\n" << "LogLike: " << _log_likelihood
  // 	    << "\nLogPrior: " << _log_prior
  // 	    << std::endl;

  return ratio;
};








template< class InnerModelType >
void outer_rlsbp<InnerModelType>::initialize_clusters(
  const stickygpm::stickygpm_regression_data<
    typename outer_rlsbp<InnerModelType>::scalar_type >& data,
  const int k
) {
  const int trunc = _MaxClust_ + 1;
  _cluster_labels.resize( data.n() );
  _reference_labels.resize( data.n() );
  _cluster_indices.resize( trunc );
  _Clustering_cost = Eigen::MatrixXi::Zero( trunc, trunc );
  //
  _reserve_cluster_indices( data.n() );
  //
  if ( k <= 1 ) {
    _initialize_single_cluster_label( data.Y() );
  }
  else {
    std::cout << "\t(k = " << k << ")\n";
    _initialize_cluster_labels( data.Y(), std::min(k, trunc) );
  }
  _shrink_cluster_indices();
  _Initialized_ = true;
};






template< class InnerModelType >
void outer_rlsbp<InnerModelType>::sort_clusters() {
  // Re-label reference clusters to place in descending-size order
  std::vector<int> ord( _cluster_indices.size() );
  std::vector<int> remap( ord.size() );
  std::vector<int> csize( _cluster_indices.size(), 0 );
  int L;
  // First count size of reference clusters
  for (int i = 0; i < (int)_reference_labels.size(); i++) {
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
  for (int k = 0; k < (int)remap.size(); k++) {
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
const InnerModelType& outer_rlsbp<InnerModelType>::inner_model_ref(
  const int which
) const {
  if ( which < 0 || which > _MaxClust_ ) {
    throw std::domain_error("Argument 'which' out of scope");
  }
  return _inner_models[which];
};




template< class InnerModelType >
void outer_rlsbp<InnerModelType>::_initialize_cluster_labels(
  const typename outer_rlsbp<InnerModelType>::matrix_type& Y,
  const int k
) {
  // const int trunc = _MaxClust_ + 1;
  // _cluster_labels.resize( Y.cols() );
  // _reference_labels.resize( Y.cols() );
  // _cluster_indices.resize( trunc );
  // _reserve_cluster_indices( Y.cols() );
  // _Clustering_cost = Eigen::MatrixXi::Zero( trunc, trunc );
  std::vector<vector_type> cluster_centers(
    k, vector_type::Zero(Y.rows()) );
  // cluster_centers.reserve( k );
  // std::vector<double> cluster_probabilities;
  scalar_type mean_y, n_sigma2_y;
  int cluster, nclust, occupied_clusters = 0;
  const scalar_type e_inv = std::exp(-1);
  _cluster_labels[0] = 0;
  // cluster_centers.push_back( Y.col(0).eval() );
  cluster_centers[0] += Y.col(0).eval();
  _cluster_indices[0].push_back(0);
  // _cluster_counts[0]++;
  occupied_clusters++;
  for ( int i = 1; i < Y.cols(); i++ ) {
    std::vector<double> cluster_probabilities( k, e_inv );
    // cluster_probabilities.clear();
    // cluster_probabilities.resize(
    //   std::min(occupied_clusters + 1, k));
    mean_y = Y.col(i).mean();
    n_sigma2_y = ( Y.col(i) -
		  vector_type::Constant(Y.rows(), mean_y) )
      .squaredNorm();
    for ( int j = 0; j < occupied_clusters; j++ ) {
      cluster_probabilities[j] = std::exp( -0.5 *
        ( Y.col(i) - cluster_centers[j] ).squaredNorm() /
					   n_sigma2_y );
      // ^^ relative probabilities
    }
    // if ( (int)cluster_probabilities.size() < k ) {
    //   cluster_probabilities.back() = e_inv;
    // }
    std::discrete_distribution<int> Categorical(
      cluster_probabilities.begin(), cluster_probabilities.end());
    cluster = Categorical( stickygpm::rng() );
    _cluster_labels[i] = cluster;
    _reference_labels[i] = cluster;
    _Clustering_cost.coeffRef(cluster, cluster) -= 1;
    _cluster_indices[cluster].push_back(i);
    nclust = _cluster_indices[cluster].size();
    if ( i < (Y.cols() - 1) ) {
      // Only update cluster centers if needed for next iteration
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
  // _shrink_cluster_indices();
  // _Initialized_ = true;
};






template< class InnerModelType >
void outer_rlsbp<InnerModelType>::_initialize_single_cluster_label(
  const typename outer_rlsbp<InnerModelType>::matrix_type& Y
) {  
  for (int i = 0; i < Y.cols(); i++) {
    _cluster_labels[i] = 0;
    _reference_labels[i] = 0;
    _Clustering_cost.coeffRef(0, 0) -= 1;
    _cluster_indices[0].push_back(i);
  }
};






template< class InnerModelType >
void outer_rlsbp<InnerModelType>::_update_cluster_labels(
  const stickygpm::stickygpm_regression_data<
    typename outer_rlsbp<InnerModelType>::scalar_type >& data,
  const typename outer_rlsbp<InnerModelType>::vector_type& sigma_sq_inv,
  const double pr_use_likelihood,
  const bool update_reference_labels,
  const bool align_to_reference
) {
  // const double eps0 = 1e-12;
  const int N = data.n();
  Eigen::VectorXd vPrk(_LoCoeffs_W.cols());
  std::vector<double> likelihood(_LoCoeffs_W.cols());
  double remaining_stick, highest_loglik;
  double sumlike;
  double log_posterior, subject_loglik;
  int cluster;
  long int max_cluster_size = 0;

  bool use_likelihood;
  std::uniform_real_distribution<> Uniform(0, 1);
  
  _log_likelihood = 0;
  _Clustering_cost.setZero( _MaxClust_ + 1, _MaxClust_ + 1 );

  // Reserve space for cluster indices
  for (int j = 0; j < (int)_cluster_indices.size(); j++) {
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
    //
    // vPrk.setZero( _LoCoeffs_W.cols() );
    // for ( typename sparse_matrix_type::InnerIterator it(data.Z(), i);
    // 	  it; ++it ) {
    //   for ( int j = 0; j < vPrk.size(); j++ ) {
    // 	vPrk.coeffRef(j) += static_cast<double>(it.value()) *
    // 	  static_cast<double>(_LoCoeffs_W.coeffRef( it.index(), j ));
    //   }
    // }
    //
    remaining_stick = 1;
    highest_loglik = 0;
    sumlike = 0;

    use_likelihood = (Uniform(stickygpm::rng()) < pr_use_likelihood);
    //
// #pragma omp parallel for shared(_inner_models, data, likelihood, sigma_sq_inv, i)
//     for ( int j = 0; j < (int)likelihood.size(); j++ ) {
//       likelihood[j] = _inner_models[j].log_kernel( data, sigma_sq_inv, i );
//       likelihood[j] = isnan(likelihood[j]) ? -HUGE_VAL : likelihood[j];
//     }
    //
    
    for (int j = 0; j < vPrk.size(); j++) {
      vPrk.coeffRef(j) = isnan( vPrk.coeffRef(j) ) ? 0 :
	extra_distributions::std_logistic_cdf( vPrk.coeffRef(j) );
      _pr_clust_i[j] = vPrk.coeffRef(j) * remaining_stick;
      remaining_stick *= ( 1 - vPrk.coeffRef(j) );

      
      if ( use_likelihood ) {	
	// first store the log likelihood
	likelihood[j] = _inner_models[j].log_kernel(
          data, sigma_sq_inv, i );
	likelihood[j] = isnan(likelihood[j]) ? -HUGE_VAL : likelihood[j];
	
	highest_loglik = (j == 0) ?
	  likelihood[j] : std::max(highest_loglik, likelihood[j]);
      }  // if (use_likelihood)
      
    }
    // end - for (int j = 0; j < vPrk.size(); j++)


    if ( use_likelihood ) {
      
      for (int j = 0; j < (int)likelihood.size(); j++) {
	likelihood[j] = std::exp(likelihood[j] - highest_loglik);
	sumlike += likelihood[j];
      }
      for (int j = 0; j < (int)_pr_clust_i.size(); j++) {
	_pr_clust_i[j] *= likelihood[j] / sumlike;
      }

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
    //
    if ( use_likelihood ) {
      // Undo previous scaling
      subject_loglik = std::log( likelihood[cluster] ) + highest_loglik;
    }
    else {
      subject_loglik = log_kernel( data, sigma_sq_inv, i );
    }
    _log_likelihood += subject_loglik;
    // _log_likelihood += log_likelihood( data, sigma_sq_inv, i );
  }
  // end - for (int i = 0; i < N; i++)  ...
  _shrink_cluster_indices();
  //
  if ( update_reference_labels ) {
    // Use only the part of the log_posterior proportional to beta
    log_posterior = _log_likelihood + _log_prior;
    if ( log_posterior > _ref_log_posterior ) {
      _ref_log_posterior = log_posterior;
      _reference_labels.assign(
        _cluster_labels.begin(), _cluster_labels.end());
    }
  }
  if ( align_to_reference ) {
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
  _log_likelihood += 0.5 * data.n() *
    sigma_sq_inv.template cast<double>().array().log().sum();
  //
};







template< class InnerModelType >
void outer_rlsbp<InnerModelType>::_update_logistic_coefficients(
  const stickygpm::stickygpm_regression_data<
    typename outer_rlsbp<InnerModelType>::scalar_type >& data
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
      
      for (int i = 0; i < (int)_cluster_labels.size(); i++) {
	cluster = _cluster_labels[i];
        mu = data.Z().row(i) * _LoCoeffs_W.col(j);
	// mu = 0;
	// for ( typename sparse_matrix_type::InnerIterator it(data.Z(), i);
	//       it; ++it ) {
	//   mu += it.value() * _LoCoeffs_W.coeffRef( it.index(), j );
	// }
	//
	mu = std::isinf( mu ) ? 100 : mu;
	if ( cluster == j ) {
	  upper = 1e3;
	  lower = 0;
	}
	else {
	  upper = 0;
	  lower = -1e3;
	}
	truncated_logistic_distribution<scalar_type> TLogis(
          mu, 1, lower, upper);
	_eta.coeffRef(i) = TLogis( stickygpm::rng() );
	residual = _eta.coeffRef(i) - mu;
	residual2 = isnan(residual) ? eps0 : (residual * residual);
	logistic_weight_distribution<scalar_type> LWD( residual2 );
	_lambda_inv.coeffRef(i) = 1 / LWD( stickygpm::rng() );
      }
      // for (int i = 0; ...  (loop over participants)
    
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
      	lower = -1e2;
      	upper = (scalar_type)extra_distributions
      	  ::std_logistic_quantile( (double)1 / (N + 1) );
      }
      else {
      	lower = (scalar_type)extra_distributions
      	  ::std_logistic_quantile( (double)N / (N + 1) );
      	upper = 1e2;
      }
        // _w_mu0.coeffRef(0),
      	// 1 / _sigma2_inv.coeffRef(0),
      _LoCoeffs_W.col(j) = vector_type::Zero( _LoCoeffs_W.rows() );
      if ( _has_intercept ) {
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
  // for (int j = 0; ...  (loop over clusters)
  // std::cout << std::endl;
};




template< class InnerModelType >
void outer_rlsbp<InnerModelType>::_update_logistic_hyperparameters(
  const std::vector<Eigen::VectorXi>& random_effects_indices,
  const std::vector<Eigen::VectorXi>& fixed_effects_indices
) {
  const int N = _cluster_labels.size();
  bool update_cluster_parameters = true;
  // Update random effects hyperparameters
  if ( !random_effects_indices.empty() ) {
    std::normal_distribution<scalar_type> Gaussian(0, 1);
    scalar_type w, sum_w, sum_w2;
    scalar_type mu, tau, tau_rate;
    int n;
    
    for ( const Eigen::VectorXi& indices : random_effects_indices ) {
      n = indices.size();
      for ( int j = 0; j < _LoCoeffs_W.cols(); j++ ) {

	/* Only update cluster hyperparameters if both of the 
	 * following hold:
	 *   a) The cluster is non-empty
	 *   b) The cluster is not fully occupied by all N samples
	 *      for an LSBP with a global intercept
	 *
	 * There are numerical difficulties updating the parameters
	 * in both of these cases: the logistic regression outcomes
	 * are either all 0 or all 1. The regressions parameters
	 * (besides the intercept) have no meaning in this context
	 */
	update_cluster_parameters = !_cluster_indices[j].empty() &&
	  !( _has_intercept && (int)_cluster_indices[j].size() == N );
	  
	if ( update_cluster_parameters ) {
	  tau = _sigma2_inv.coeffRef( indices.coeffRef(0), j );
	  sum_w = 0;
	  sum_w2 = 0;
	  for ( int i = 0; i < n; i++ ) {
	    w = _LoCoeffs_W.coeffRef( indices.coeffRef(i), j );
	    sum_w += w;
	    sum_w2 += w * w;
	  }
	  if ( _has_intercept ) {    // ***
	    // ^^ This condition is not complete for all cases
	    // but will do for now
	    mu = static_cast<scalar_type>( 0 );
	  }
	  else {
	    mu = sum_w / n + std::sqrt( 1 / (n * tau) ) *
	      Gaussian( stickygpm::rng() );
	  }
	  tau_rate = 0.5 * (sum_w2 - 2 * mu * sum_w + n * mu * mu) + 1;

	  std::gamma_distribution<scalar_type> Gamma(
            0.5 * n + 1, 1 / tau_rate
          );
	  tau = Gamma( stickygpm::rng() );

	  for ( int i = 0; i < n; i++ ) {
	    _w_mu0.coeffRef( indices.coeffRef(i), j ) = mu;
	    _sigma2_inv.coeffRef( indices.coeffRef(i), j ) = tau;
	  }
	}  // if ( update_cluster_parameters )
      }  // for ( int j = 0; j < _LoCoeffs_W.cols(); j++ )
    }  // for ( const Eigen::VectorXi& indices : random_effects_indices )
    
  }
  // if ( !random_effects_indices.empty() )

  
  /*
   * Update fixed effects hyperparameters (horseshoe)
   */
  const Eigen::VectorXi& fei = fixed_effects_indices[0];
  if ( fei.size() > 0 ) {
    scalar_type nui, xii;
    
    for ( int j = 0; j < _loc_shrink.cols(); j++ ) {

      update_cluster_parameters = !_cluster_indices[j].empty() &&
	!( _has_intercept && (int)_cluster_indices[j].size() == N );
      
      if ( update_cluster_parameters ) {
	scalar_type sum_ls = 0;
	scalar_type gs = _glb_shrink.coeff(j);
      
	for ( int i_fei = 0; i_fei < fei.size(); i_fei++ ) {
	
	  int i = fei[i_fei];
	  scalar_type ls = _loc_shrink.coeff(i, j);
	  scalar_type Wij_sq = _LoCoeffs_W.coeff(i, j) *
	    _LoCoeffs_W.coeff(i, j);
	
	  std::gamma_distribution<scalar_type>
	    Gamma_nu( 1, 1 / (1 + ls) );
	  nui = Gamma_nu( stickygpm::rng() );
	
	  std::gamma_distribution<scalar_type>
	    Gamma_ls( 1, 1 / (nui + Wij_sq * gs / 2) );
	  ls = Gamma_ls( stickygpm::rng() );
	  ls = isnan( ls ) ?
	    std::numeric_limits<scalar_type>::max() : ls;
	  ls = std::min(ls, std::numeric_limits<scalar_type>::max());
	
	  //
	  sum_ls += Wij_sq * ls;
	  _loc_shrink.coeffRef(i, j) = ls;
	}
      
	std::gamma_distribution<scalar_type>
	  Gamma_xi( 1, 1 / (1 + gs) );
	xii = Gamma_xi( stickygpm::rng() );
      
	std::gamma_distribution<scalar_type>
	  Gamma_gs( (fei.size() + 1) / 2,
		    1 / (xii + sum_ls / 2) );
	gs = Gamma_gs( stickygpm::rng() );
	gs = std::min(gs, std::numeric_limits<scalar_type>::max());
	//
	_glb_shrink.coeffRef(j) = gs;
	//
	for ( int i_fei = 0; i_fei < fei.size(); i_fei++ ) {
	  int i = fei[i_fei];
	  _sigma2_inv.coeffRef( i, j ) = 
	    ( _loc_shrink.coeff(i, j) * _glb_shrink.coeff(j) );
	}
      }  // if ( update_cluster_parameters )
    }    // for ( int j = 0; ...
  }      // if ( fei.size() > 0 )
};





template< class InnerModelType >
void outer_rlsbp<InnerModelType>::_draw_gaussian() {
  std::normal_distribution<scalar_type> Gaussian(0, 1);
  for (int i = 0; i < _rgauss.size(); i++) {
    _rgauss[i] = Gaussian( stickygpm::rng() );
  }
};




template< class InnerModelType >
void outer_rlsbp<InnerModelType>::_reorder_clusters(
  const std::vector<int>& new_labels
) {
  std::vector<int> ord(new_labels.cbegin(), new_labels.cend());
  // ^^ Modified in keeping track of remaining indices to be ordered
  vector_type wtemp(_LoCoeffs_W.rows());
  scalar_type xtemp;
  // Reassign cluster labels
  for ( int& cl : _cluster_labels ) {
    cl = new_labels[ cl ];
  }
  //
  // Reorder cluster:
  //   atoms / indices / logistic coefficients
  for (int k = 0; k < (int)new_labels.size(); k++) {
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
    wtemp = _sigma2_inv.col(k);
    _sigma2_inv.col(k) = _sigma2_inv.col(ord[k]);
    _sigma2_inv.col(ord[k]) = wtemp;
    //
    wtemp = _w_mu0.col(k);
    _w_mu0.col(k) = _w_mu0.col(ord[k]);
    _w_mu0.col(ord[k]) = wtemp;
    //
    wtemp = _loc_shrink.col(k);
    _loc_shrink.col(k) = _loc_shrink.col(ord[k]);
    _loc_shrink.col(ord[k]) = wtemp;
    //
    xtemp = _glb_shrink.coeff(k);
    _glb_shrink.coeffRef(k) = _glb_shrink.coeff(ord[k]);
    _glb_shrink.coeffRef(ord[k]) = xtemp;
    //
    // Update 'ord'
    for (int h = k; h < (int)ord.size(); h++) {
      if ( ord[h] == k ) {
	ord[h] = ord[k];
	break;
      }
    }
  }
  //
};




template< class InnerModelType >
void outer_rlsbp<InnerModelType>::_reserve_cluster_indices(
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
void outer_rlsbp<InnerModelType>::_shrink_cluster_indices() {
  typedef std::vector< std::vector<int> >::iterator vit;
  for (vit it = _cluster_indices.begin();
       it != _cluster_indices.end(); ++it) {
    it->shrink_to_fit();
  }
};




#endif  // _STICKYGPM_OUTER_RLSBP_

