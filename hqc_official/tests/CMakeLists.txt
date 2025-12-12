cmake_minimum_required(VERSION 3.21)
include(CTest)

# Must be passed in from top-level
if(NOT DEFINED HQC_ARCH)
    message(FATAL_ERROR "HQC_ARCH must be set (ref or x86_64)")
endif()

# -------------------------
# external MUnit framework
# -------------------------
add_library(munit STATIC
    ${CMAKE_SOURCE_DIR}/tests/external/munit/munit.c
)
target_include_directories(munit PUBLIC
    ${CMAKE_SOURCE_DIR}/tests/external/munit
)

# Force C11 for munit (to support _Atomic) and disable extensions
set_target_properties(munit PROPERTIES
    C_STANDARD           11
    C_STANDARD_REQUIRED  ON
    C_EXTENSIONS         OFF
)

# Disable warnings-as-errors and specific warnings in munit
target_compile_options(munit PRIVATE
    -Wno-error
    -Wno-newline-eof
    $<$<C_COMPILER_ID:Clang>:-Wno-c11-extensions>
    $<$<C_COMPILER_ID:GNU>:-Wno-gnu>
)

set(VARIANTS "hqc-1;hqc-3;hqc-5" CACHE INTERNAL "List of HQC variants")

# Where KAT diffs and verbose outputs go
set(TEST_OUTPUT_DIR "${CMAKE_BINARY_DIR}/test_outputs")
file(MAKE_DIRECTORY "${TEST_OUTPUT_DIR}")

add_subdirectory(unit)
add_subdirectory(api)
add_subdirectory(kats)
add_subdirectory(bench)
