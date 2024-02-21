cmake_minimum_required(VERSION 3.0.0)
project(fcc_qp_solver VERSION 0.1.0)
if (APPLE)
set(CMAKE_CXX_FLAGS "-std=c++11 -stdlib=libc++")
endif()
add_subdirectory(pybind11)

list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_LIST_DIR}/eigen/cmake")
set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/eigen")
find_package(Eigen3 3.3.90 EXACT REQUIRED)

find_package(SPQR REQUIRED)
find_package(Cholmod REQUIRED)

set(SOURCE_DIR "src")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

set(SRCS
    ${SOURCE_DIR}/fcc_qp.cpp
    )

set(HEADERS
    ${SOURCE_DIR}/fcc_qp.hpp
    )

include_directories(${EIGEN3_INCLUDE_DIR} ${CHOLMOD_INCLUDES} ${SPQR_INCLUDES})

add_library(fcc_qplib ${SRCS} ${HEADERS})
target_link_libraries(fcc_qplib pybind11::pybind11 ${CHOLMOD_LIBRARIES} ${SPQR_LIBRARIES})
target_compile_options(fcc_qplib PRIVATE -fPIC)

pybind11_add_module(fcc_qp_solver "${SOURCE_DIR}/main.cpp")
target_link_libraries(fcc_qp_solver PRIVATE fcc_qplib)
target_compile_options(fcc_qp_solver PRIVATE -fPIC)

