cmake_minimum_required(VERSION 3.0.0)
project(conehead_solver VERSION 0.1.0)
if (APPLE)
set(CMAKE_CXX_FLAGS "-std=c++11 -stdlib=libc++")
endif()
add_subdirectory(pybind11)
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_LIST_DIR}/eigen/cmake")
set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/eigen")
find_package(Eigen3 3.3.90 EXACT REQUIRED)
set(SOURCE_DIR "src")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

set(SRCS
    ${SOURCE_DIR}/conehead_solver.cpp
    )
    
set(HEADERS
    ${SOURCE_DIR}/conehead_solver.hpp
    )

include_directories(${EIGEN3_INCLUDE_DIR})

add_library(coneheadlib ${SRCS} ${HEADERS})
target_link_libraries(coneheadlib pybind11::pybind11)
target_compile_options(coneheadlib PRIVATE -fPIC)

pybind11_add_module(conehead_solver "${SOURCE_DIR}/main.cpp")
target_link_libraries(conehead_solver PRIVATE coneheadlib)
target_compile_options(conehead_solver PRIVATE -fPIC)

