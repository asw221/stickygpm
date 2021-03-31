
find_package(Boost COMPONENTS system filesystem REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
## find_package(FFTW REQUIRED)
find_package(NLOPT REQUIRED)
find_package(OpenMP REQUIRED)
find_package(ZLIB REQUIRED)


if (NOT TARGET OpenMP::OpenMP_CXX)
  message(FATAL_ERROR "OpenMP_CXX target not found")
endif()





include_directories(${stickygpm_SOURCE_DIR}/include)
include_directories(${stickygpm_SOURCE_DIR}/lib/abseil/include)
include_directories(${stickygpm_SOURCE_DIR}/lib/nifti/include)

link_directories(${stickygpm_SOURCE_DIR}/lib/nifti/lib)




## --- covest --------------------------------------------------------
add_executable(covest covest.cpp)
target_link_libraries(covest Boost::filesystem Eigen3::Eigen
		NLOPT::nlopt OpenMP::OpenMP_CXX
		libniftiio.a libznz.a ZLIB::ZLIB)
set_target_properties(covest PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
		


## --- image_knots ---------------------------------------------------
add_executable(image_knots image_knots.cpp)
target_link_libraries(image_knots Boost::filesystem Eigen3::Eigen
		libniftiio.a libznz.a ZLIB::ZLIB)
set_target_properties(image_knots PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")


## --- sgpreg --------------------------------------------------------
add_executable(sgpreg sgpreg.cpp)
target_link_libraries(sgpreg Boost::filesystem Eigen3::Eigen
		NLOPT::nlopt OpenMP::OpenMP_CXX
		libniftiio.a libznz.a ZLIB::ZLIB)
set_target_properties(sgpreg PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")


## --- threshold_nifti -----------------------------------------------
add_executable(threshold_nifti threshold_nifti.cpp)
target_link_libraries(threshold_nifti Boost::filesystem Eigen3::Eigen
		libniftiio.a libznz.a ZLIB::ZLIB)
set_target_properties(threshold_nifti PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")


## ## --- dualgpm -------------------------------------------------------
## add_executable(dualgpm dualgpm.cpp)
## target_link_libraries(dualgpm Boost::filesystem FFTW::FloatOpenMP
## 		FFTW::Float FFTW::DoubleOpenMP FFTW::Double
## 		Eigen3::Eigen NLOPT::nlopt OpenMP::OpenMP_CXX
## 		libniftiio.a libznz.a ZLIB::ZLIB)
## set_target_properties(dualgpm PROPERTIES RUNTIME_OUTPUT_DIRECTORY
## 		"${CMAKE_BINARY_DIR}/bin")
## target_compile_definitions(dualgpm PUBLIC DUALRES_DONT_UPDATE_TAU)
## target_compile_definitions(dualgpm PUBLIC EIGEN_DONT_PARALLELIZE)


