
#include <Eigen/Core>
#include <vector>



template< class ArgType, class RowIndexType, class ColIndexType >
Eigen::CwiseNullaryOp<
  stickygpm::matrix_indexing_functor<ArgType, RowIndexType, ColIndexType>,
  typename stickygpm::matrix_indexing_functor<
    ArgType, RowIndexType, ColIndexType>::MatrixType>
stickygpm::nullary_index(
  const Eigen::MatrixBase<ArgType> &arg,
  const RowIndexType &row_indices,
  const ColIndexType& col_indices
) {
  typedef stickygpm::matrix_indexing_functor<
    ArgType, RowIndexType, ColIndexType> Func;
  typedef typename Func::MatrixType MatrixType;
  return MatrixType::NullaryExpr(
    row_indices.size(), col_indices.size(),
    Func(arg.derived(), row_indices, col_indices)
  );
};

    

template< class ArgType, class IndexType >
Eigen::CwiseNullaryOp<
  stickygpm::vector_indexing_functor<ArgType, IndexType>,
  typename stickygpm::vector_indexing_functor<ArgType, IndexType>::VectorType >
stickygpm::nullary_index(
  const Eigen::MatrixBase<ArgType> &arg,
  const IndexType &row_indices
) {
  typedef stickygpm::vector_indexing_functor<ArgType, IndexType> Func;
  typedef typename Func::VectorType VectorType;
  return VectorType::NullaryExpr(row_indices.size(), Func(arg.derived(), row_indices));
};






template< typename MatrixType >
MatrixType stickygpm::eigen_select(
  const MatrixType &M,
  const std::vector<int> &row_indices,
  const std::vector<int> &col_indices
) {
  MatrixType Sub(row_indices.size(), col_indices.size());
  int i, j = 0;
  for (std::vector<int>::const_iterator jt = col_indices.cbegin();
       jt != col_indices.cend(); ++jt, ++j) {
    i = 0;
    for (std::vector<int>::const_iterator it = row_indices.cbegin();
	 it != row_indices.cend(); ++it, ++i) {
      Sub(i, j) = M(*it, *jt);
    }
  }
  return Sub;
};



template< typename MatrixType >
MatrixType stickygpm::eigen_select_symmetric(
  const MatrixType &M, const std::vector<int> &indices
) {
  MatrixType Sub(indices.size(), indices.size());
  int i, j = 0;
  // Loop only over lower triangle + diagonal
  for (std::vector<int>::const_iterator jt = indices.cbegin();
       jt != indices.end(); ++jt, ++j) {
    i = j;
    for (std::vector<int>::const_iterator it = jt;
	 it != indices.cend(); ++it, ++i) {
      Sub(i, j) = M(*it, *jt);
      Sub(j, i) = M(*it, *jt);  // redundant assignment for diagonal
    }
  }
  return Sub;
};
