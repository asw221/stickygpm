
cmake_minimum_required(VERSION 3.12)
project(stickygpm)


set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
# set(CMAKE_CXX_FLAGS_RELEASE "-O2 -Wall -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -Wall")
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")



add_subdirectory(src)



if (APPLE)
   set(CMAKE_C_COMPILER clang)
   set(CMAKE_CXX_COMPILER clang++)

   # Hack for Xcode 12.3:
   link_directories(/usr/lib)
   
   if(CMAKE_C_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_C "${CMAKE_C_COMPILER}")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp=libomp -Wno-unused-command-line-argument")
        set(OpenMP_C_LIB_NAMES "omp" "gomp" "iomp5")
        set(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES})
        set(OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES})
        set(OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES})
   endif()
   if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
      set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
      set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp=libomp -Wno-unused-command-line-argument")
      set(OpenMP_CXX_LIB_NAMES "omp" "gomp" "iomp5")
      set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
      set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
      set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
   endif()

   # if (CMAKE_C_COMPILER_ID MATCHES "Clang\$")
   #    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
   #    set(OpenMP_C_LIB_NAMES "omp")
   #    set(OpenMP_omp_LIBRARY omp)
   # endif()
   # if (CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
   #    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
   #    set(OpenMP_CXX_LIB_NAMES "omp")
   #    set(OpenMP_omp_LIBRARY omp)
   # endif()

endif()



if (CMAKE_CXX_COMPILER_ID MATCHES "GNU\$")
    # require at least gcc 8.0
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0)
        message(FATAL_ERROR "GCC version must be >= 8.0")
    endif()
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
    # require at least clang 6.0
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0)
        message(FATAL_ERROR "Clang version must be >= 6.0")
    endif()
else()
    message(WARNING "You are using an unsupported compiler! Compilation has only been tested with clang and gcc")
endif()

