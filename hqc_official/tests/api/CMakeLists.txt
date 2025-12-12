cmake_minimum_required(VERSION 3.21)
include(CTest)

file(GLOB API_TEST_SRCS
        "${CMAKE_CURRENT_SOURCE_DIR}/test_*.c"
)

set(API_MAIN_SRC "${CMAKE_CURRENT_SOURCE_DIR}/tests_main.c")

foreach(variant IN LISTS VARIANTS)
    string(REPLACE "-" "_" safe_var ${variant})
    set(exe "test_api_${safe_var}")

    add_executable(${exe}
            ${API_MAIN_SRC}
            ${API_TEST_SRCS}
    )

    target_link_libraries(${exe} PRIVATE
            munit
            ${safe_var}_${HQC_ARCH}
            fips202
    )

    target_compile_definitions(${exe} PRIVATE
            $<$<STREQUAL:${HQC_ARCH},x86_64>:AVX2_MODE>
            $<$<NOT:$<STREQUAL:${HQC_ARCH},x86_64>>:REF_MODE>
    )

    target_include_directories(${exe} PRIVATE
            ${CMAKE_SOURCE_DIR}/src/common
            ${CMAKE_SOURCE_DIR}/src/${HQC_ARCH}/${variant}
            ${CMAKE_SOURCE_DIR}/lib/fips202
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/tests
            ${CMAKE_SOURCE_DIR}/tests/external/munit
    )

    add_test(NAME ${exe} COMMAND ${exe})
    set_tests_properties(${exe} PROPERTIES LABELS "api")
endforeach()
