
#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>


#ifndef _STICKYGPM_MEDIAN_
#define _STICKYGPM_MEDIAN_


template< typename Iterator >
typename std::iterator_traits<Iterator>::value_type
median( Iterator first, Iterator last ) {
  const int n = std::abs(std::distance(first, last));
  if ( n == 0 ) return *first;
  std::vector<size_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
	    [&](size_t a, size_t b) -> bool {
	      return *(first + a) < *(first + b);
	    });
  if (n % 2 == 0)
    return (*(first + order[n/2]) + *(first + order[n/2 - 1])) / 2;
  return *(first + order[n/2]);
};


/* Median absolute deviation */
template< typename Iterator >
typename std::iterator_traits<Iterator>::value_type
mad( Iterator first, Iterator last, const double k = 1.482602 ) {
  typedef typename std::iterator_traits<Iterator>::value_type
    value_type;
  const int n = std::abs(std::distance(first, last));
  const value_type center = median(first, last);
  std::vector<value_type> centered_data(n);
  size_t i = 0;
  for ( Iterator it = first; it != last; ++it, ++i )
    centered_data[i] = std::abs(*it - center);
  return median(centered_data.begin(), centered_data.end()) *
    static_cast<value_type>(k);
};


#endif  // _STICKYGPM_MEDIAN_
