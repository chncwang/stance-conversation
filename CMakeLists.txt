CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(single-turn-conversation)

SET(CMAKE_CXX_COMPILER g++)

function(find_mkl)
  set(MKL_ARCH intel64)
  set(MKL_ROOT /opt/intel/mkl)
  find_path(MKL_INCLUDE_DIR mkl.h
            PATHS ${MKL_ROOT} ${MKL_ROOT}/include)
  find_library(MKL_CORE_LIB NAMES mkl_intel_lp64 mkl_sequential mkl_core
               PATHS ${MKL_ROOT} ${MKL_ROOT}/lib/${MKL_ARCH}
               DOC "MKL core library path")

  find_library(MKL_COMPILER_LIB NAMES iomp5 libiomp5md
               PATHS ${MKL_ROOT} ${MKL_ROOT}/../compiler/lib/${MKL_ARCH}              #Windows
                     ${MKL_ROOT}/../compilers_and_libraries/linux/lib/${MKL_ARCH}_lin #Linux
               DOC "MKL compiler lib (for threaded MKL)")

  if(MKL_INCLUDE_DIR AND MKL_CORE_LIB AND MKL_COMPILER_LIB)
    get_filename_component(MKL_CORE_LIB_DIR ${MKL_CORE_LIB} DIRECTORY)
    get_filename_component(MKL_COMPILER_LIB_DIR ${MKL_COMPILER_LIB} DIRECTORY)
    get_filename_component(MKL_COMPILER_LIB_FILE ${MKL_COMPILER_LIB} NAME)
    message(STATUS "Found MKL\n   * include: ${MKL_INCLUDE_DIR},\n   * core library dir: ${MKL_CORE_LIB_DIR},\n   * compiler library: ${MKL_COMPILER_LIB}")

    # Due to a conflict with /MT and /MD, MSVC needs mkl_intel_lp64 linked last, or we can change individual
    # projects to use /MT (mkl_intel_lp64 linked with /MT, default MSVC projects use /MD), or we can instead
    # link to the DLL versions. For now I'm opting for this solution which seems to work with projects still
    # at their default /MD. Linux build requires the mkl_intel_lp64 to be linked first. So...:
    if(MSVC)
      set(LIBS ${LIBS} mkl_sequential mkl_core mkl_intel_lp64 ${MKL_COMPILER_LIB_FILE} PARENT_SCOPE)
    else()
      set(LIBS ${LIBS} mkl_intel_lp64 mkl_sequential mkl_core ${MKL_COMPILER_LIB_FILE} PARENT_SCOPE)
    endif()
    include_directories(${MKL_INCLUDE_DIR})
    link_directories(${MKL_CORE_LIB_DIR} ${MKL_COMPILER_LIB_DIR})
    set(MKL_LINK_DIRS ${MKL_CORE_LIB_DIR} ${MKL_COMPILER_LIB_DIR} PARENT_SCOPE) # Keeping this for python build
  else()
    message(FATAL_ERROR "Failed to find MKL in path: ${MKL_ROOT} (Did you set MKL_ROOT properly?)")
  endif()
endfunction()

######## Cross-compiler, cross-platform options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_FAST_MATH -Wall -Wno-sign-compare")
if (MKL OR MKL_ROOT)
  find_mkl()  # sets include/lib directories and sets ${LIBS} needed for linking
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_USE_MKL_ALL")
endif()

ADD_DEFINITIONS(-DDEVICE_MEMORY=2)

if(CMAKE_BUILD_TYPE MATCHES debug)
    ADD_DEFINITIONS(-DN3LDG_DEBUG)
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -msse3 -std=c++11 -O1" )
else()
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC -std=c++11 -O3 -march=native" )
endif()

SET(Boost_USE_STATIC_LIBS OFF) 
SET(Boost_USE_MULTITHREADED ON)  
SET(Boost_USE_STATIC_RUNTIME OFF) 
FIND_PACKAGE(Boost REQUIRED COMPONENTS regex system filesystem) 
MESSAGE("boost includes:" ${Boost_INCLUDE_DIRS})
MESSAGE("boost include:" ${Boost_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS}) 
INCLUDE_DIRECTORIES(${EIGEN})
INCLUDE_DIRECTORIES(3rd/jsoncpp)
INCLUDE_DIRECTORIES(3rd/n3ldg-plus/include 3rd/cxxopts/include src)
INCLUDE_DIRECTORIES(3rd/inih)
INCLUDE_DIRECTORIES(3rd/tinyutf8)

AUX_SOURCE_DIRECTORY(3rd/jsoncpp SRCS)
AUX_SOURCE_DIRECTORY(src/single_turn_conversation SRCS)
ADD_EXECUTABLE(main ${SRCS})

IF(TEST_CUDA)
    ADD_DEFINITIONS(-DTEST_CUDA)
ENDIF()

IF(USE_DOUBLE)
    ADD_DEFINITIONS(-DUSE_DOUBLE)
ENDIF()

# IF(USE_CUDA)
#     ADD_DEFINITIONS(-DUSE_GPU)
#     INCLUDE_DIRECTORIES(3rd/n3ldg-plus/cuda)
#     INCLUDE_DIRECTORIES(/usr/local/cuda/samples/common/inc)
#     INCLUDE_DIRECTORIES(cnmem/include cuda include)
# ENDIF()
IF(USE_CUDA)
set(CUDA_SEPARABLE_COMPILATION ON)
list(APPEND CUDA_NVCC_FLAGS "--use_fast_math;-arch=compute_${CARD_NUMBER};-code=sm_${CARD_NUMBER};-std=c++11;-DVERBOSE;-DEIGEN_USE_GPU;-DHAVE_CUDA;")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
if (CMAKE_BUILD_TYPE MATCHES debug)
    list(APPEND CUDA_NVCC_FLAGS "-G")
    list(APPEND CUDA_NVCC_FLAGS "-g")
    list(APPEND CUDA_NVCC_FLAGS "-O1")
else()
    list(APPEND CUDA_NVCC_FLAGS "-O3")
endif()
ADD_DEFINITIONS(-DUSE_GPU)
INCLUDE_DIRECTORIES(/usr/local/cuda/samples/common/inc)
INCLUDE_DIRECTORIES(3rd/n3ldg-plus/cnmem/include 3rd/n3ldg-plus/cuda 3rd/n3ldg-plus/include)
FIND_PACKAGE(CUDA REQUIRED)
FIND_CUDA_HELPER_LIBS(curand)
SET(CUDA_TOOLKIT_ROOT_DIR ${CUDA_ROOT})
MESSAGE("CUDA_INCLUDE: " ${CUDA_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(SYSTEM ${CUDA_INCLUDE_DIRS})
SET(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})
MESSAGE("CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
SET(LIBS ${LIBS} ${CUDA_LIBRARIES})
ADD_SUBDIRECTORY(3rd/n3ldg-plus/cnmem)
CUDA_ADD_LIBRARY(n3ldg_cuda 3rd/n3ldg-plus/cuda/N3LDG_cuda.cu)
CUDA_ADD_CUBLAS_TO_TARGET(n3ldg_cuda)
TARGET_LINK_LIBRARIES(n3ldg_cuda cnmem)
SET(LIBS ${LIBS} n3ldg_cuda)
ENDIF()
ADD_SUBDIRECTORY(3rd/n3ldg-plus)

TARGET_LINK_LIBRARIES(main pthread ${LIBS})
TARGET_LINK_LIBRARIeS(main ${Boost_LIBRARIES})
