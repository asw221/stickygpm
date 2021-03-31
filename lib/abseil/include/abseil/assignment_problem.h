
#include <algorithm>
#include <cassert>
#include <cmath>
#include <Eigen/Core>
#include <limits>
#include <stdexcept>
#include <vector>


/*
 * Implements the complete cost matrix primal-dual algorithm
 * (CTC algorithm) from:
 * 
 * Carpaneto, G., & Toth, P. (1987). Primal-dual algrorithms for the 
 *   assignment problem. Discrete Applied Mathematics, 18(2), 137-153.
 */



#ifndef _ABSEIL_MIN_COST_ASSIGNMENT_
#define _ABSEIL_MIN_COST_ASSIGNMENT_



namespace abseil {


  class min_cost_assignment {

  public:
    template< typename T, int _Rows, int _Cols, int _Options,
	      int _MaxCompTimeRows, int _MaxCompTimeCols >
    static void solve(
      const Eigen::Matrix<T, _Rows, _Cols, _Options,
      _MaxCompTimeRows, _MaxCompTimeCols>& Cost,
      std::vector<int>& column_labels       // Carpaneto & Toth : fbar
    );

    
    template< typename T, int _Rows, int _Cols, int _Options,
	      int _MaxCompTimeRows, int _MaxCompTimeCols >
    static void solve(
      const Eigen::Matrix<T, _Rows, _Cols, _Options,
      _MaxCompTimeRows, _MaxCompTimeCols>& Cost,
      std::vector<int>& column_labels,      // Carpaneto & Toth : fbar
      T& min_cost
    );

    

  private:
    static const int _UNASSIGNED;

    template< typename T, int _Rows, int _Cols, int _Options,
	      int _MaxCompTimeRows, int _MaxCompTimeCols >
    static T compute_min_cost(
      const Eigen::Matrix<T, _Rows, _Cols, _Options,
      _MaxCompTimeRows, _MaxCompTimeCols>& Cost,
      std::vector<int>& col_lab
    );

    template< typename T >
    static bool equiv(const T& x, const T& y);
    // ^^ Re-write this with SFINAE at some point
    
    template< typename T, int _Rows, int _Cols, int _Options,
	      int _MaxCompTimeRows, int _MaxCompTimeCols >
    static void initialize(
      const Eigen::Matrix<T, _Rows, _Cols, _Options,
      _MaxCompTimeRows, _MaxCompTimeCols>& Cost,
      std::vector<int>& column_labels,
      std::vector<int>& row_lab_f,
      std::vector<T>& U,
      std::vector<T>& V,
      int& m_assigned
     );

    template< typename T, int _Rows, int _Cols, int _Options,
	      int _MaxCompTimeRows, int _MaxCompTimeCols >
    static int min_row_index(
      const Eigen::Matrix<T, _Rows, _Cols, _Options,
      _MaxCompTimeRows, _MaxCompTimeCols>& Cost,
      const int& col,
      const std::vector<int>& tiebreak
    );

    template< typename T, int _Rows, int _Cols, int _Options,
	      int _MaxCompTimeRows, int _MaxCompTimeCols >    
    static int min_col_index(
      const Eigen::Matrix<T, _Rows, _Cols, _Options,
      _MaxCompTimeRows, _MaxCompTimeCols>& Cost,
      const int& row,
      const std::vector<T>& offset,
      const std::vector<int>& tiebreak
    );
    
  };
  // class min_cost_assignment



  
}
// namespace abseil





const int abseil::min_cost_assignment::_UNASSIGNED = -1;






template< typename T, int _Rows, int _Cols, int _Options,
	  int _MaxCompTimeRows, int _MaxCompTimeCols >
void abseil::min_cost_assignment::solve(
  const Eigen::Matrix<T, _Rows, _Cols, _Options,
  _MaxCompTimeRows, _MaxCompTimeCols>& Cost,
  std::vector<int>& column_labels           // Carpaneto & Toth : fbar
) {
  const int N = Cost.rows();
  std::vector<int> row_lab_f,             // C&T : f
    alternate_row_start_c(N, 0),          // C&T : C
    unlabeled_col(N, 0),                  // C&T : UC
    ucol_ind_ic(N, 0);                    // C&T : IC
  int m_assigned;                         // C&T : m
  std::vector<T> alternate_cost(N, 0),    // C&T : pi
    U, V;
  int xswap, g, i, j, k, s, W;
  int r = 0;
  T D;
  bool flag;
  if ( N != Cost.cols() ) {
    throw std::domain_error(
      "min_cost_assignment: Cost matrix must be square");
  }
  initialize(Cost, column_labels, row_lab_f, U, V, m_assigned);
  while (m_assigned < N) {
    if ( row_lab_f[r] == _UNASSIGNED ) {
      // Search for an alternate/augmenting path starting from row r
      for (j = 0; j < N; j++) {
	alternate_row_start_c[j] = r;
	unlabeled_col[j] = j;
	alternate_cost[j] = Cost.coeffRef(r, j) - U[r] - V[j];
      }
      W = N;
      g = -1;
      while ( g < 0 ) {
	// Compute D = min{ alternate_cost[j] : Col j is unlabeled }
	D = std::numeric_limits<T>::max();
	// g = -1;
	k = 0;
	s = 0;
	flag = false;
	while ( k < W  &&  !flag ) {
	  j = unlabeled_col[k];
	  if ( alternate_cost[j] <= D ) {
	    if ( alternate_cost[j] < D ) {
	      g = -1;
	      s = 0;
	      D = alternate_cost[j];
	    }
	    if ( column_labels[j] == _UNASSIGNED ) {
	      g = j;
	      // flag = (D == 0);
	      flag = equiv(D, 0);
	    }
	    ucol_ind_ic[s] = k;
	    s++;
	  }
	  k++;
	}
	// end : while ( k < W  &&  !flag )
	if ( g == -1 ) {
	  for (int q = (s - 1); q >= 0; q--) {
	    k = ucol_ind_ic[q];
	    xswap = unlabeled_col[k];
	    W--;
	    unlabeled_col[k] = unlabeled_col[W];
	    unlabeled_col[W] = xswap;
	    i = std::max(column_labels[xswap], 0);
	    for (int t = 0; t < W; t++) {
	      j = unlabeled_col[t];
	      if ( alternate_cost[j] >
		   (D + Cost.coeffRef(i, j) - U[i] - V[j]) ) {
		alternate_cost[j] =
		  D + Cost.coeffRef(i, j) - U[i] - V[j];
		alternate_row_start_c[j] = i;
	      }
	    }
	  }
	}
      }
      // end : while ( g < 0 )
      // Update dual variables:
      for (k = W; k < N; k++) {
	j = unlabeled_col[k];
	i = column_labels[j];
	V[j] += alternate_cost[j] - D;
	U[i] -= alternate_cost[j] - D;
      }
      U[r] += D;
      // Assign new row
      while ( i != r ) {
	i = alternate_row_start_c[g];
	column_labels[g] = i;
	xswap = row_lab_f[i];
	row_lab_f[i] = g;
	g = xswap;
      }
      m_assigned++;
    }
    // end : if ( row_lab_f[r] == _UNASSIGNED )
    r++;
  }
  // end : while (m_assigned < N)
};







template< typename T, int _Rows, int _Cols, int _Options,
	  int _MaxCompTimeRows, int _MaxCompTimeCols >
void abseil::min_cost_assignment::solve(
  const Eigen::Matrix<T, _Rows, _Cols, _Options,
  _MaxCompTimeRows, _MaxCompTimeCols>& Cost,
  std::vector<int>& column_labels,          // Carpaneto & Toth : fbar
  T& min_cost
) {
  solve(Cost, column_labels);
  min_cost = compute_min_cost(Cost, column_labels);
};




template< typename T, int _Rows, int _Cols, int _Options,
	  int _MaxCompTimeRows, int _MaxCompTimeCols >
void abseil::min_cost_assignment::initialize(
  const Eigen::Matrix<T, _Rows, _Cols, _Options,
  _MaxCompTimeRows, _MaxCompTimeCols>& Cost,
  std::vector<int>& column_labels,
  std::vector<int>& row_lab_f,
  std::vector<T>& U,
  std::vector<T>& V,
  int& m_assigned
) {
  // Search for (partial) initial solution
  const int N = Cost.rows();
  std::vector<int> unscanned_col_p(N, 0);  // C&T : p
  int r, i, j, k;
  bool flag;
  m_assigned = 0;
  column_labels.assign(N, _UNASSIGNED);
  row_lab_f.assign(N, _UNASSIGNED);
  U.assign(N, 0);
  V.assign(N, 0);
  // Initialize column dual variables V
  for (j = 0; j < N; j++) {
    r = min_row_index(Cost, j, row_lab_f);
    V[j] = Cost.coeffRef(r, j);
    if ( row_lab_f[r] == _UNASSIGNED ) {
      m_assigned++;
      column_labels[j] = r;
      row_lab_f[r] = j;
      unscanned_col_p[r] = j + 1;
    }
  }
  // end : for (j = 0; j < N; j++)
  for (i = 0; i < N; i++) {
    if ( row_lab_f[i] == _UNASSIGNED ) {
      j = min_col_index(Cost, i, V, column_labels);
      U[i] = Cost.coeffRef(i, j) - V[j];
      flag = false;
      while ( row_lab_f[i] != _UNASSIGNED  &&  j < N ) {
	// if ( (Cost.coeffRef(i, j) - U[i] - V[j]) == 0 ) {
	if ( equiv(Cost.coeffRef(i, j) - U[i] - V[j], 0) ) {
	  r = column_labels[j];
	  k = unscanned_col_p[r];
	  while ( !flag  &&  k < N ) {
	    if ( column_labels[k] == _UNASSIGNED  &&
		 equiv(Cost.coeffRef(r, k) - U[r] - V[k], 0) ) {
		 // (Cost.coeffRef(r, k) - U[r] - V[k]) == 0 ) {
	      flag = true;
	    }
	    else {
	      k++;
	    }
	  }
	  unscanned_col_p[r] = k + 1;
	}
	if ( flag ) {
	  column_labels[j] = 0;
	  row_lab_f[r] = k;
	  column_labels[k] = r;
	}
      }
      // end : while ( row_lab_f[i] != _UNASSIGNED  &&  j < N )
      if ( column_labels[j] == _UNASSIGNED ) {
	m_assigned++;
	row_lab_f[i] = j;
	column_labels[j] = i;
	unscanned_col_p[i] = j + 1;
      }
    }
    // end : if ( row_lab_f[i] == _UNASSIGNED )
  }
  // end : for (i = 0; i < N; i++)
};









template< typename T, int _Rows, int _Cols, int _Options,
	  int _MaxCompTimeRows, int _MaxCompTimeCols >
T abseil::min_cost_assignment::compute_min_cost(
  const Eigen::Matrix<T, _Rows, _Cols, _Options,
  _MaxCompTimeRows, _MaxCompTimeCols>& Cost,
  std::vector<int>& column_labels
) {
  assert(Cost.cols() == column_labels.size() &&
	 "Incorrect cost dimensions");
  T total_cost = 0;
  for (int j = 0; j < Cost.cols(); j++) {
    total_cost += Cost.coeffRef(column_labels[j], j);
  }
  return total_cost;
};





template< typename T >
bool abseil::min_cost_assignment::equiv(
  const T& x, const T& y
) {
  const T tol = 1e-4;
  return std::abs(x - y) <= tol;
};


template<>
bool abseil::min_cost_assignment::equiv<int>(
  const int& x, const int& y
) {
  return x == y;
};




template< typename T, int _Rows, int _Cols, int _Options,
	  int _MaxCompTimeRows, int _MaxCompTimeCols >
int abseil::min_cost_assignment::min_row_index(
  const Eigen::Matrix<T, _Rows, _Cols, _Options,
  _MaxCompTimeRows, _MaxCompTimeCols>& Cost,
  const int& col,
  const std::vector<int>& tiebreak
) {
  const int N = Cost.rows();
  int imin = 0;
  for (int i = 1; i < N; i++) {
    if ( Cost.coeffRef(i, col) < Cost.coeffRef(imin, col) ) {
      imin = i;
    }
    else if ( equiv(Cost.coeffRef(i, col), Cost.coeffRef(imin, col)) ) {
    // else if ( Cost.coeffRef(i, col) == Cost.coeffRef(imin, col) ) {
      if ( tiebreak[imin] != _UNASSIGNED  &&
	   tiebreak[i] == _UNASSIGNED ) {
	imin = i;
      }
    }
  }
  return imin;
};



template< typename T, int _Rows, int _Cols, int _Options,
	  int _MaxCompTimeRows, int _MaxCompTimeCols >
int abseil::min_cost_assignment::min_col_index(
  const Eigen::Matrix<T, _Rows, _Cols, _Options,
  _MaxCompTimeRows, _MaxCompTimeCols>& Cost,
  const int& row,
  const std::vector<T>& offset,
  const std::vector<int>& tiebreak
) {
  const int M = Cost.cols();
  int jmin = 0;
  T C, Cmin = Cost.coeffRef(row, jmin) - offset[jmin];
  for (int j = 1; j < M; j++) {
    C = Cost.coeffRef(row, j) - offset[j];
    if ( C < Cmin ) {
      jmin = j;
      Cmin = C;
    }
    else if ( equiv(C, Cmin) ) {
    // else if ( C == Cmin ) {
      if ( tiebreak[jmin] != _UNASSIGNED  &&
	   tiebreak[j] == _UNASSIGNED ) {
	jmin = j;
	Cmin = C;
      }
    }
  }
  return jmin;
};








    
//    template< typename T, int _Rows, int _Cols, int _Options,
//      int _MaxCompTimeRows, int _MaxCompTimeCols >
//     static std::vector<int> inits(
//       const Eigen::Matrix<T, _Rows, _Cols, _Options,
//       _MaxCompTimeRows, _MaxCompTimeCols>& Cost
//     );
//
//
// template< typename T, int _Rows, int _Cols, int _Options,
//   int _MaxCompTimeRows, int _MaxCompTimeCols >
// std::vector<int> abseil::min_cost_assignment::inits(
//   const Eigen::Matrix<T, _Rows, _Cols, _Options,
//   _MaxCompTimeRows, _MaxCompTimeCols>& Cost
// ) {
//   const int N = Cost.rows();
//   std::vector<int> column_labels,   // Carpaneto & Toth : fbar
//     row_lab_f;                     // C&T : f
//   int m_assigned;                  // C&T : m
//   std::vector<T> U, V;
//   if ( N != Cost.cols() ) {
//     throw std::domain_error(
//       "min_cost_assignment: Cost matrix must be square");
//   }
//   initialize(Cost, column_labels, row_lab_f, U, V, m_assigned);
//   return column_labels;
// };






#endif  // _ABSEIL_MIN_COST_ASSIGNMENT_
