cmake_minimum_required (VERSION 3.13)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
project (TF_KNN LANGUAGES CUDA CXX)

option(BUILD_GPU "Enable building the tensorflow operator"
       ON)
option(BUILD_TEST "Enable building the tensorflow operator"
       ON)
option(BUILD_TF_OP "Enable building the tensorflow operator"
       ON)

# The version number.
set (TF_KNN_VERSION_MAJOR 0)
set (TF_KNN_VERSION_MINOR 1)
set (CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS_DEBUG_INIT "-Wall")
#set(CMAKE_CXX_FLAGS_RELEASE_INIT "-Wall")

include(CheckLanguage)
check_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Set CUDA architectures for compatibility with different GPUs
# Including compute capability 8.6 for RTX 3090
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "60;61;70;75;80;86")
endif()

set(CUDA_NVCC_FLAGS_DEBUG "-g -G")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

#Search for python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

#TODO: Windows...
#list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11" )

set(COMMON_SOURCE_LIST
    "kdtree.cpp"
    "tf_nndistance_g.cu"
    "tf_kdtree_g.cu")

set(COMMON_HEADERS_LIST
    "cutils.cuh"
    "kdtree.hpp"
    "nndistance.hpp")

#set(EXEC_SOURCE_LIST
#	"${COMMON_SOURCE_LIST}"
#    "test_kdtree.cpp")

set(EXEC_HEADERS_LIST
    "${COMMON_HEADERS_LIST}")
    
set(INTERFACE_SOURCE_LIST
    "${COMMON_SOURCE_LIST}"
    "tf_nndistance_g.cu"
    "tf_nndistance.cpp"
    "tf_kdtree.cpp")


include(ExternalProject)
set(EIGEN_INSTALL_DIR "${PROJECT_SOURCE_DIR}/eigenlib")
ExternalProject_Add(eigen
  GIT_REPOSITORY  "https://gitlab.com/libeigen/eigen"
  GIT_TAG 3.4
  SOURCE_DIR "${EIGEN_INSTALL_DIR}"
  INSTALL_COMMAND ""
  BUILD_COMMAND ""
  CONFIGURE_COMMAND ""
  )

#Old, see below
#INCLUDE_DIRECTORIES("${EIGEN_INSTALL_DIR}")


##############################################################
## Tensorflow includes and libraries
execute_process(COMMAND "${Python_EXECUTABLE}" "-c" "import tensorflow as tf; import sys; ret = sys.stdout.write(tf.sysconfig.get_lib() + '/')" 
                OUTPUT_VARIABLE TF_LIB_DIR 
                RESULT_VARIABLE TF_LIB_RET_VAL)
execute_process(COMMAND "${Python_EXECUTABLE}" "-c" "import tensorflow as tf; import sys; ret = sys.stdout.write(tf.sysconfig.get_include() + '/')" 
                OUTPUT_VARIABLE TF_INCLUDE_DIR
                RESULT_VARIABLE TF_INCLUDE_RET_VAL) 
#TODO: Error checking not properly working
#TODO: Calling python_executable not working

if(NOT TF_LIB_RET_VAL EQUAL 0 OR NOT TF_INCLUDE_RET_VAL EQUAL 0)
    message( FATAL_ERROR "Error: Importing tensorflow in the active python environment failed")
endif()

#https://stackoverflow.com/questions/28597351/how-do-i-add-a-library-path-in-cmake
link_directories(${TF_LIB_DIR})

set(CONDA_CUDART_LIBRARY "$ENV{CONDA_PREFIX}/lib/libcudart.so.11.0")
if(EXISTS "${CONDA_CUDART_LIBRARY}")
    set(CUDART_LIBRARY "${CONDA_CUDART_LIBRARY}")
elseif(EXISTS "$ENV{CONDA_PREFIX}/lib/libcudart.so")
    set(CUDART_LIBRARY "$ENV{CONDA_PREFIX}/lib/libcudart.so")
else()
    set(CUDART_LIBRARY cudart)
endif()

##############################################################
## Create targets and add definitions, dependencies...

#if (BUILD_TEST)
#    # add the executable
#    add_executable(test_kdtree ${EXEC_SOURCE_LIST}) # ${EXEC_HEADERS_LIST})
#endif()

#PYTHON_ADD_MODULE(oum_export_release tsitsiklis.cpp OrderedUpwindMethod.cpp utility.cpp OUMNPInterface.cpp)

#set_target_properties(test_kdtree PROPERTIES
#                            CUDA_SEPARABLE_COMPILATION ON)

add_library(tf_nndistance SHARED ${INTERFACE_SOURCE_LIST})

if(EXISTS "$ENV{CONDA_PREFIX}/lib")
    target_link_directories(tf_nndistance PRIVATE "$ENV{CONDA_PREFIX}/lib")
endif()

target_compile_definitions(tf_nndistance PRIVATE _GLIBCXX_USE_CXX11_ABI=0)

#target_link_libraries(test_kdtree ${CUDART_LIBRARY})
target_link_libraries(tf_nndistance tensorflow_framework ${CUDART_LIBRARY})

#target_include_directories(test_kdtree PRIVATE ${EIGEN_INSTALL_DIR})
target_include_directories(tf_nndistance PRIVATE ${EIGEN_INSTALL_DIR} ${TF_INCLUDE_DIR})

#-fPIC
#set_property(tf_nndistance PROPERTY POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(tf_nndistance PUBLIC "$<$<CONFIG:RELEASE>:NDEBUG>")
#target_compile_definitions(test_kdtree PUBLIC "$<$<CONFIG:RELEASE>:NDEBUG>")

##################################
# Post-build commands
add_custom_command(TARGET tf_nndistance 
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tf_nndistance> "${PROJECT_SOURCE_DIR}/..")

##################################
# Tests
enable_testing()
add_test (NAME python-tests
  COMMAND "${Python_EXECUTABLE}" #-m pytest 
  ${PROJECT_SOURCE_DIR}/../scripts/test_knn_unit.py
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/../"
)

#Dependencies
add_dependencies(tf_nndistance eigen)
#add_dependencies(test_kdtree eigen)
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND}
                  DEPENDS tf_nndistance)


##################################
# Debug: Print all variables
get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()