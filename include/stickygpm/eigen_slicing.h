
#include <Eigen/Core>
#include <vector>





// *_indexing_functor's from:
// http://eigen.tuxfamily.org/dox-devel/TopicCustomizing_NullaryExpr.html#title1



#ifndef _STICKYGPM_EIGEN_SLICING_
#define _STICKYGPM_EIGEN_SLICING_


namespace stickygpm {

      
    template< typename MatrixType >
    MatrixType eigen_select(
      const MatrixType &M,
      const std::vector<int> &row_indices,
      const std::vector<int> &col_indices
    );


    template< typename MatrixType >
    MatrixType eigen_select_symmetric(const MatrixType &M, const std::vector<int> &indices);




    template< class ArgType, class RowIndexType, class ColIndexType >
    class matrix_indexing_functor {
    private:
      const ArgType &m_arg;
      const RowIndexType &m_rowIndices;
      const ColIndexType &m_colIndices;
  
    public:
      typedef Eigen::Matrix<typename ArgType::Scalar,
        RowIndexType::SizeAtCompileTime,
        ColIndexType::SizeAtCompileTime,
        ArgType::Flags & Eigen::RowMajorBit ? Eigen::RowMajor : Eigen::ColMajor,
        RowIndexType::MaxSizeAtCompileTime,
        ColIndexType::MaxSizeAtCompileTime> MatrixType;
      
      matrix_indexing_functor(
        const ArgType &arg,
	const RowIndexType &row_indices,
	const ColIndexType& col_indices
      ) : m_arg(arg), m_rowIndices(row_indices), m_colIndices(col_indices)
      { ; }
  
      const typename ArgType::Scalar& operator() (Eigen::Index row, Eigen::Index col) const {
	return m_arg(m_rowIndices[row], m_colIndices[col]);
      };
    };



    template< class ArgType, class IndexType >
    class vector_indexing_functor {
    private:
      const ArgType &m_arg;
      const IndexType &m_Indices;
  
    public:
      typedef Eigen::Matrix<typename ArgType::Scalar,
        IndexType::SizeAtCompileTime,
        1,
        ArgType::Flags & Eigen::RowMajorBit ? Eigen::RowMajor : Eigen::ColMajor,
        IndexType::MaxSizeAtCompileTime,
        1> VectorType;
      
      vector_indexing_functor(
        const ArgType &arg,
	const IndexType &row_indices
      ) : m_arg(arg), m_Indices(row_indices)
      { ; }
  
      const typename ArgType::Scalar& operator() (Eigen::Index ind) const {
	return m_arg(m_Indices[ind]);
      };
    };




  
    
  template< class ArgType, class RowIndexType, class ColIndexType >
  Eigen::CwiseNullaryOp<
    stickygpm::matrix_indexing_functor<ArgType, RowIndexType, ColIndexType>,
    typename stickygpm::matrix_indexing_functor<
      ArgType, RowIndexType, ColIndexType>::MatrixType>
  nullary_index(
    const Eigen::MatrixBase<ArgType> &arg,
    const RowIndexType &row_indices,
    const ColIndexType& col_indices
  );

    

  template< class ArgType, class IndexType >
  Eigen::CwiseNullaryOp<
    stickygpm::vector_indexing_functor<ArgType, IndexType>,
    typename stickygpm::vector_indexing_functor<ArgType, IndexType>::VectorType >
  nullary_index(
    const Eigen::MatrixBase<ArgType> &arg,
    const IndexType &row_indices
  );



  
}  // namespace stickygpm


#include "stickygpm/impl/eigen_slicing.inl"


#endif  // _STICKYGPM_EIGEN_SLICING_

