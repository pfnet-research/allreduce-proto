project("ibcomm")

# CMake version
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_CXX_FLAGS "-Wno-missing-field-initializers -Wno-format-security -Wno-sign-compare")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting CMAKE_BUILD_TYPE=Release (default)")
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(MPI)
find_package(CUDA)

# ibverbs
find_library(IBVERBS_LIBRARY
  NAMES ibverbs libibverbs libibverbs.so
  HINTS ENV LD_LIBRARY_PATH)

find_path(IBVERBS_INCLUDE_PATH
  NAMES "infiniband/verbs.h"
  HINTS ENV CPATH)

# google test
find_library(GOOGLETEST_MAIN_LIBRARY
  NAMES gtest_main libgtest_main libgtest_main.a
  HINTS
  ${GOOGLETEST_ROOT}
  ${GOOGLETEST_ROOT}/build
  ${GOOGLETEST_ROOT}/build/googlemock/gtest
  )

find_library(GOOGLETEST_LIBRARY
  NAMES gtest libgtest libgtest.a
  HINTS
  ${GOOGLETEST_ROOT}
  ${GOOGLETEST_ROOT}/build
  ${GOOGLETEST_ROOT}/build/googlemock/gtest
  )

find_path(GOOGLETEST_INCLUDE_PATH
  NAMES "gtest/gtest.h"
  HINTS
  ${GOOGLETEST_ROOT}
  ${GOOGLETEST_ROOT}/include
  ${GOOGLETEST_ROOT}/googletest/include)

message(STATUS "GOOGLETEST_ROOT=${GOOGLETEST_ROOT}")
message(STATUS "GOOGLETEST_LIBRARY=${GOOGLETEST_LIBRARY}")
message(STATUS "GOOGLETEST_INCLUDE_PATH=${GOOGLETEST_INCLUDE_PATH}")

# Run mpicxx -show to get compile flags for MPI
set(CMAKE_CXX_COMPILER "${CMAKE_SOURCE_DIR}/mpinvcc.sh")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter")

include_directories(".")

# for libraries
if(CUDA_FOUND)
    add_library(ibcomm_cuda SHARED
        ibcomm/ibverbs_communicator.cpp
        ibcomm/ibverbs_communicator_cuda.cpp
        ibcomm/memory_pool.cpp
        )
    if(USE_TRACE)
        target_compile_definitions(ibcomm_cuda
            PUBLIC "-DUSE_CUDA"
            PUBLIC "-DUSE_TRACE"
            )
    else()
        target_compile_definitions(ibcomm_cuda
            PUBLIC "-DUSE_CUDA"
            )
    endif()

    target_include_directories(ibcomm_cuda
        PUBLIC ${IBVERBS_INCLUDE_PATH}
        PUBLIC ${CUDA_INCLUDE_DIRS}
        )
    target_link_libraries(ibcomm_cuda
        ${IBVERBS_LIBRARY}
        ${CUDA_LIBRARIES}
        )
endif()

add_library(ibcomm SHARED ibcomm/ibverbs_communicator.cpp)
if(USE_TRACE)
    target_compile_definitions(ibcomm
        PUBLIC "-DUSE_TRACE"
        )
endif()
target_include_directories(ibcomm PUBLIC ${IBVERBS_INCLUDE_PATH})
target_link_libraries(ibcomm ${IBVERBS_LIBRARY})

# for tests
add_executable(sendrecv tests/sendrecv_test.cpp)
target_link_libraries(sendrecv ibcomm)

if(GOOGLETEST_INCLUDE_PATH)
  add_executable(unittest tests/unittest.cpp)
  target_link_libraries(unittest ${GOOGLETEST_LIBRARY} ${GOOGLETEST_MAIN_LIBRARY})
  target_include_directories(unittest PUBLIC ${GOOGLETEST_INCLUDE_PATH})
  target_include_directories(unittest PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endif()

add_library(tinyexpr tinyexpr/tinyexpr.c)

if(CUDA_FOUND)
    add_executable(allreduce_tester ibcomm/allreduce_tester.cpp)
    target_include_directories(allreduce_tester PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/tinyexpr)
    target_include_directories(allreduce_tester PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/grumpi)
    target_link_libraries(allreduce_tester ibcomm_cuda)
    target_link_libraries(allreduce_tester tinyexpr)
endif()

# for allreduce examples
add_executable(allreduce examples/allreduce.cpp)
target_link_libraries(allreduce ibcomm)
set_target_properties(allreduce PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples"
)

if(CUDA_FOUND)
    add_executable(allreduce_cuda examples/allreduce.cpp)
    target_compile_definitions(allreduce_cuda
        PUBLIC "-DUSE_CUDA"
    )
    target_link_libraries(allreduce_cuda ibcomm_cuda)
    set_target_properties(allreduce_cuda PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples"
    )
endif()
