cmake_minimum_required(VERSION 3.10)
project(StochasticSwarm
    VERSION 1.1.0
    DESCRIPTION "High-performance particle simulation with Langevin dynamics"
    LANGUAGES CXX
)

# Project options
option(BUILD_TESTS        "Build unit tests"            ON)
option(ENABLE_WARNINGS    "Enable compiler warnings"    ON)
option(WARNINGS_AS_ERRORS "Treat warnings as errors"    OFF)

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# Compiler warnings
# ============================================================================
if(ENABLE_WARNINGS)
    if(MSVC)
        add_compile_options(/W4)
        if(WARNINGS_AS_ERRORS)
            add_compile_options(/WX)
        endif()
    else()
        add_compile_options(-Wall -Wextra -Wpedantic)
        if(WARNINGS_AS_ERRORS)
            add_compile_options(-Werror)
        endif()
    endif()
endif()

# ============================================================================
# Release optimisation flags
# ============================================================================
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(NOT MSVC)
        add_compile_options(-O3 -march=native -ffast-math)
    endif()
endif()

# ============================================================================
# OpenMP
# ----------------------------------------------------------------------------
# Apple's system Clang ships without libomp.  When `brew install libomp` is
# present we manually point CMake at the right headers + dylib before calling
# find_package(OpenMP) so it succeeds on Apple Silicon / macOS.
# ============================================================================
if(APPLE)
    # Locate Homebrew's libomp prefix (works with both Intel and Apple Silicon)
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    if(HOMEBREW_LIBOMP_PREFIX)
        message(STATUS "Found Homebrew libomp: ${HOMEBREW_LIBOMP_PREFIX}")
        # Tell CMake's OpenMP finder the right flags / library
        set(OpenMP_CXX_FLAGS
            "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY
            "${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib")
    endif()
endif()

find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP ${OpenMP_CXX_VERSION} enabled -- parallel particle loop active")
else()
    message(STATUS "OpenMP NOT found -- simulation will run single-threaded")
    message(STATUS "  macOS: brew install libomp")
    message(STATUS "  Linux: apt install libomp-dev  (or equivalent)")
endif()

# ============================================================================
# Include path
# ============================================================================
include_directories(${PROJECT_SOURCE_DIR}/include)

# ============================================================================
# Main executable
# ============================================================================
add_executable(${PROJECT_NAME} src/main.cpp)

if(OpenMP_CXX_FOUND)
    # OpenMP::OpenMP_CXX imported target carries compile + link flags automatically
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
endif()

file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/output)

# ============================================================================
# Python bindings
# ============================================================================
find_package(Python3 COMPONENTS Interpreter Development QUIET)

if(Python3_FOUND)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    if(pybind11_DIR)
        find_package(pybind11 CONFIG QUIET PATHS "${pybind11_DIR}")

        if(pybind11_FOUND)
            message(STATUS "Found pybind11: ${pybind11_VERSION}")
            message(STATUS "Python: ${Python3_EXECUTABLE}")

            pybind11_add_module(stochastic_swarm bindings/bindings.cpp)

            target_include_directories(stochastic_swarm PRIVATE include)

            # NOTE: Do NOT link OpenMP into the Python extension.
            # On macOS (and often Linux) numpy/scipy already load libomp, and linking
            # a second copy into the .so causes "libomp already initialized" crash.
            # The #pragma omp directives in the headers are silently ignored by clang
            # when compiled without -fopenmp, so the extension runs single-threaded
            # but crash-free.  Full OpenMP is active in the standalone C++ executables.

            set_target_properties(stochastic_swarm PROPERTIES
                LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            )

            message(STATUS "Python bindings will be built")
        else()
            message(STATUS "pybind11 CMake config not found")
        endif()
    else()
        message(STATUS "pybind11 not found - Python bindings disabled")
        message(STATUS "Install with: pip install pybind11")
    endif()
else()
    message(STATUS "Python3 not found - Python bindings disabled")
endif()

# ============================================================================
# Tests + benchmarks
# ============================================================================
if(BUILD_TESTS)
    enable_testing()

    # Helper macro: add a test/benchmark target and optionally link OpenMP
    macro(add_ss_executable target source)
        add_executable(${target} ${source})
        if(OpenMP_CXX_FOUND)
            target_link_libraries(${target} OpenMP::OpenMP_CXX)
        endif()
    endmacro()

    add_ss_executable(test_rng            tests/test_rng.cpp)
    add_ss_executable(test_particle_system tests/test_particle_system.cpp)
    add_ss_executable(benchmark_spatial_hash           tests/benchmark_spatial_hash.cpp)
    add_ss_executable(benchmark_spatial_hash_optimized tests/benchmark_spatial_hash_optimized.cpp)
    add_ss_executable(benchmark_msd               tests/benchmark_msd.cpp)
    add_ss_executable(benchmark_particle_system   tests/benchmark_particle_system.cpp)

    add_test(NAME RNG_Tests           COMMAND test_rng)
    add_test(NAME ParticleSystem_Tests COMMAND test_particle_system)

    add_custom_target(run_tests
        COMMAND ${CMAKE_CTEST_COMMAND} --verbose
        DEPENDS test_rng test_particle_system
        COMMENT "Running all tests..."
    )

    add_custom_target(run_benchmarks
        COMMAND benchmark_spatial_hash
        COMMAND benchmark_spatial_hash_optimized
        COMMAND benchmark_msd
        COMMAND benchmark_particle_system
        DEPENDS
            benchmark_spatial_hash
            benchmark_spatial_hash_optimized
            benchmark_msd
            benchmark_particle_system
        COMMENT "Running all C++ performance benchmarks..."
    )
endif()

# ============================================================================
# Installation
# ============================================================================
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
install(FILES README.md DESTINATION share/doc/${PROJECT_NAME})

# ============================================================================
# Configuration summary
# ============================================================================
message(STATUS "============================================================")
message(STATUS "  StochasticSwarm Build Configuration")
message(STATUS "============================================================")
message(STATUS "  Version:         ${PROJECT_VERSION}")
message(STATUS "  Build Type:      ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:    ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler:        ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Build Tests:     ${BUILD_TESTS}")
message(STATUS "  OpenMP:          ${OpenMP_CXX_FOUND}")
message(STATUS "  Warnings:        ${ENABLE_WARNINGS}")
message(STATUS "  Install Prefix:  ${CMAKE_INSTALL_PREFIX}")
message(STATUS "============================================================")
