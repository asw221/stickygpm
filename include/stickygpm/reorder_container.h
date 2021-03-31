
#include <algorithm>
#include <iterator>


#ifndef _STICKYGPM_REORDER_CONTAINER_
#define _STICKYGPM_REORDER_CONTAINER_


namespace stickygpm {
  

template< typename value_iterator, typename order_iterator >
void reorder(
  value_iterator v,
  order_iterator order_begin,
  order_iterator order_end
) {
  typedef typename std::iterator_traits<value_iterator>
    ::value_type value_t;
  typedef typename std::iterator_traits<order_iterator>
    ::value_type index_t;
  typedef typename std::iterator_traits<order_iterator>
    ::difference_type diff_t;
    
  diff_t remaining = order_end - order_begin - 1;
  value_t temp;
  for (index_t s = index_t(), d; remaining > 0; ++s) {
    for (d = order_begin[s]; d > s; d = order_begin[d])
      { ; }
    if ( d == s ) {
      --remaining;
      temp = v[s];
      while ( d = order_begin[d], d != s ) {
	std::swap(temp, v[d]);
	--remaining;
      }
      v[s] = temp;
    }
  }
};


}

#endif  // _STICKYGPM_REORDER_CONTAINER_
