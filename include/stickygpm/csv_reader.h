
#include <fstream>
// #include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>



#ifndef _STICKYGPM_CSV_READER_
#define _STICKYGPM_CSV_READER_


namespace stickygpm {

  template< typename T >
  class csv_reader {
    
  public:
    static std::vector< std::vector<T> > read_file(
      const std::string filename,
      const char delimeter = ',',
      const std::string comment = "#"
    );
  
  
  private:
    static bool _parse_line(
      std::istringstream& line,
      std::vector<T>& data,
      const char delimeter
    );

    static bool _is_comment_line(
      const std::string& line,
      const std::string& comment
    );
  
  };
  // class csv_reader


}






template<>
bool stickygpm::csv_reader<std::string>::_parse_line(
  std::istringstream& line,
  std::vector<std::string>& data,
  const char delimeter
) {
  while ( line ) {
    std::string atom;
    if ( std::getline(line, atom, delimeter) ) {
      data.push_back( atom );
    }
  }
  return !data.empty();
};


template< typename T >
bool stickygpm::csv_reader<T>::_parse_line(
  std::istringstream& line,
  std::vector<T>& data,
  const char delimeter
) {
  while ( line ) {
    std::string atom;
    if ( std::getline(line, atom, delimeter) ) {
      try {
	data.push_back( (T)std::stod(atom) );
      }
      catch (...) { ; }  // <- add to later?
    }
  }
  return !data.empty();
};



template< typename T >
bool stickygpm::csv_reader<T>::_is_comment_line(
  const std::string& line,
  const std::string &comment
) {
  return line.substr(0, comment.size()) == comment;
};



template< typename T >
std::vector< std::vector<T> >
stickygpm::csv_reader<T>::read_file(
  const std::string filename,
  const char delimeter,
  const std::string comment
) {
  std::ifstream ifile( filename.c_str(), std::ifstream::in );
  std::vector< std::vector<T> > data;
  int lineno = 0;
  if ( ifile ) {
    while ( ifile ) {
      std::string line;
      lineno++;
      if ( std::getline(ifile, line) &&
	   !_is_comment_line(line, comment)) {
	std::istringstream liness(line);
	std::vector<T> line_data;
	if ( _parse_line(liness, line_data, delimeter) ) {
	  data.push_back( line_data );
	}
      }
    }  // while ( ifile )
  }
  else {
    std::string msg = "Could not open file: ";
    throw std::runtime_error( msg + filename );
  }
  ifile.close();
  return data;
};



#endif  // _STICKYGPM_CSV_READER_


