# Find NLOPT
#
# This sets the following variables:
#   NLOPT_FOUND
#   NLOPT_INCLUDE_DIRS
#   NLOPT_LIBRARIES
#   NLOPT_DEFINITIONS
#   NLOPT_VERSION
#
# Defines the following targets:
#   NLOPT::nlopt


if( NOT NLOPT_ROOT AND DEFINED ENV{NLOPTDIR} )
  set( NLOPT_ROOT $ENV{NLOPTDIR} )
endif()

# Check if we can use PkgConfig
find_package(PkgConfig QUIET)

if( PKG_CONFIG_FOUND AND NOT NLOPT_ROOT )
  pkg_check_modules( PKG_NLOPT QUIET "nlopt" )
endif()


# NLOPT installs with dynamic libs
set( CMAKE_FIND_LIBRARY_SUFFIXES_SAVED ${CMAKE_FIND_LIBRARY_SUFFIXES} )
set( CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_SHARED_LIBRARY_SUFFIX} )

if( NLOPT_ROOT )

  find_library(
    NLOPT_LIB
    NAMES "nlopt"
    PATHS ${NLOPT_ROOT}
    PATH_SUFFIXES "lib" "lib64"
    NO_DEFAULT_PATH
  )

  find_path(
    NLOPT_INCLUDE_DIRS
    NAMES "nlopt.h" "nlopt.hpp"
    PATHS ${NLOPT_ROOT}
    PATH_SUFFIXES "include"
    NO_DEFAULT_PATH
  )

else()

  find_library(
    NLOPT_LIB
    NAMES "nlopt"
    PATHS ${PKG_NLOPT_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
  )

  find_path(
    NLOPT_INCLUDE_DIRS
    NAMES "nlopt.h" "nlopt.hpp"
    PATHS ${PKG_NLOPT_LIBRARY_DIRS} ${INCLUDE_INSTALL_DIR}
  )

endif( NLOPT_ROOT )


if( NLOPT_LIB )
  set(NLOPT_LIB_FOUND TRUE)
  set(NLOPT_LIBRARIES ${NLOPT_LIBRARIES} ${NLOPT_LIB})
  add_library(NLOPT::nlopt INTERFACE IMPORTED)
  set_target_properties(NLOPT::nlopt
    PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${NLOPT_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${NLOPT_LIB}"
  )
else()
  set(NLOPT_LIB_FOUND FALSE)
endif()


# Definitions
set(NLOPT_DEFINITIONS ${PC_NLOPT_CFLAGS_OTHER})

# Include directories
find_path(NLOPT_INCLUDE_DIRS
    NAMES nlopt.h
    HINTS ${PC_NLOPT_INCLUDEDIR}
    PATHS "${CMAKE_INSTALL_PREFIX}/include")

# Libraries
find_library(NLOPT_LIBRARIES
    NAMES nlopt nlopt_cxx
    HINTS ${PC_NLOPT_LIBDIR})

# Version
set(NLOPT_VERSION ${PC_NLOPT_VERSION})


# Set (NAME)_FOUND if all the variables and the version are satisfied.
set( CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES_SAVED} )

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(NLOPT
    FAIL_MESSAGE  DEFAULT_MSG
    REQUIRED_VARS NLOPT_INCLUDE_DIRS NLOPT_LIBRARIES
    VERSION_VAR   NLOPT_VERSION)

mark_as_advanced(
  NLOPT_INCLUDE_DIRS
  NLOPT_LIBRARIES
  NLOPT_LIB
)
    