cmake_minimum_required(VERSION 3.21)
include(CTest)

if(NOT DEFINED HQC_ARCH)
    message(FATAL_ERROR "HQC_ARCH must be set (ref or x86_64)")
endif()

set(TEST_OUTPUT_DIR "${CMAKE_BINARY_DIR}/test_outputs")
file(MAKE_DIRECTORY "${TEST_OUTPUT_DIR}")

if(HQC_ARCH STREQUAL "x86_64")
    set(IMPL_DIR       "${CMAKE_SOURCE_DIR}/src/${HQC_ARCH}/${HQC_X86_IMPL}")
    set(ARCH_COMMON_DIR "${CMAKE_SOURCE_DIR}/src/${HQC_ARCH}/common")
    set(KAT_GOLD_DIR    "${CMAKE_SOURCE_DIR}/kats/${HQC_ARCH}/${HQC_X86_IMPL}")
    set(IMD_GOLD_DIR    "${CMAKE_SOURCE_DIR}/kats/${HQC_ARCH}/${HQC_X86_IMPL}")
else()
    set(IMPL_DIR       "${CMAKE_SOURCE_DIR}/src/ref")
    set(ARCH_COMMON_DIR "${CMAKE_SOURCE_DIR}/src/ref")
    set(KAT_GOLD_DIR    "${CMAKE_SOURCE_DIR}/kats/${HQC_ARCH}")
    set(IMD_GOLD_DIR    "${CMAKE_SOURCE_DIR}/kats/${HQC_ARCH}")
endif()

file(GLOB KAT_SRCS
        "${CMAKE_CURRENT_SOURCE_DIR}/test_kat.c"
)
file(GLOB IMD_SRCS
        "${CMAKE_CURRENT_SOURCE_DIR}/test_intermediates_values.c"
)

foreach(variant IN LISTS VARIANTS)
    string(REPLACE "-" "_" safe_var ${variant})

    #─────────────────────────────────────────────────────
    # 1) KAT tests
    #─────────────────────────────────────────────────────
    foreach(src IN LISTS KAT_SRCS)
        get_filename_component(name ${src} NAME_WE)
        set(exe "${name}_${safe_var}")

        add_executable(${exe} ${src})
        target_compile_options(${exe} PRIVATE
                -Wno-unused-result
                -Wno-error=unused-result
        )
        target_link_libraries(${exe} PRIVATE
                ${safe_var}_${HQC_ARCH}
                fips202
        )
        target_compile_definitions(${exe} PRIVATE
                $<$<STREQUAL:${HQC_ARCH},ref>:REF_MODE>
                $<$<AND:$<STREQUAL:${HQC_ARCH},x86_64>,$<STREQUAL:${HQC_X86_IMPL},avx256>>:AVX2_MODE>
        )
        target_include_directories(${exe} PRIVATE
                ${CMAKE_SOURCE_DIR}/tests/kats
                ${CMAKE_SOURCE_DIR}/src/common
                ${CMAKE_SOURCE_DIR}/src/common/${variant}
                ${ARCH_COMMON_DIR}
                ${ARCH_COMMON_DIR}/${variant}
                ${IMPL_DIR}/${variant}
                ${CMAKE_SOURCE_DIR}/lib/fips202
                ${CMAKE_SOURCE_DIR}/lib/nist-rng
        )

        file(GLOB GOLD_RSP
                "${KAT_GOLD_DIR}/${variant}/PQCkemKAT_*.rsp"
        )
        list(GET GOLD_RSP 0 GOLD_FILE)
        get_filename_component(GOLD_NAME ${GOLD_FILE} NAME)

        add_test(NAME ${exe}
                COMMAND /bin/sh -c
                "mkdir -p '${TEST_OUTPUT_DIR}/${exe}' && cd '${TEST_OUTPUT_DIR}/${exe}' && \
               '$<TARGET_FILE:${exe}>' && \
               diff -u '${GOLD_FILE}' '${GOLD_NAME}'"
        )
        set_tests_properties(${exe} PROPERTIES LABELS "kats")
    endforeach()

    #─────────────────────────────────────────────────────
    # 2) Intermediate-values tests
    #─────────────────────────────────────────────────────
    foreach(src IN LISTS IMD_SRCS)
        get_filename_component(name ${src} NAME_WE)
        set(exe "${name}_${safe_var}")

        file(GLOB COMMON_SRCS
                "${CMAKE_SOURCE_DIR}/src/common/*.c"
                "${CMAKE_SOURCE_DIR}/src/common/*/*.c"
        )

        file(GLOB IMPL_SRCS
                "${IMPL_DIR}/*.c"
        )
        file(GLOB ARCH_COMMON_SRCS
                "${ARCH_COMMON_DIR}/*.c"
                "${ARCH_COMMON_DIR}/${variant}/*.c"
        )
        file(GLOB VARIANT_SRCS
                "${IMPL_DIR}/${variant}/*.c"
                "${IMPL_DIR}/${variant}/*/*.c"
        )

        add_executable(${exe}
                ${src}
                ${COMMON_SRCS}
                ${ARCH_COMMON_SRCS}
                ${IMPL_SRCS}
                ${VARIANT_SRCS}
        )
        target_compile_options(${exe} PRIVATE
                -Wno-unused-result
                -Wno-error=unused-result
                -Wno-missing-prototypes
        )
        target_compile_definitions(${exe} PRIVATE
                VERBOSE
                $<$<STREQUAL:${HQC_ARCH},ref>:REF_MODE>
                $<$<AND:$<STREQUAL:${HQC_ARCH},x86_64>,$<STREQUAL:${HQC_X86_IMPL},avx256>>:AVX2_MODE>
        )
        target_include_directories(${exe} PRIVATE
                ${CMAKE_SOURCE_DIR}/tests/kats
                ${CMAKE_SOURCE_DIR}/src/common
                ${CMAKE_SOURCE_DIR}/src/common/${variant}
                ${ARCH_COMMON_DIR}
                ${ARCH_COMMON_DIR}/${variant}
                ${IMPL_DIR}
                ${IMPL_DIR}/${variant}
                ${CMAKE_SOURCE_DIR}/lib/fips202
                ${CMAKE_SOURCE_DIR}/lib/nist-rng
        )
        target_link_libraries(${exe} PRIVATE
                fips202
        )

        set(GOLD_FILE_IMD
                "${IMD_GOLD_DIR}/${variant}/intermediates_values"
        )
        if(NOT EXISTS "${GOLD_FILE_IMD}")
            message(FATAL_ERROR
                    "Could not find golden intermediates file: ${GOLD_FILE_IMD}"
            )
        endif()
        get_filename_component(GOLD_IMD_NAME ${GOLD_FILE_IMD} NAME)

        add_test(NAME ${exe}
                COMMAND /bin/sh -c
                "mkdir -p '${TEST_OUTPUT_DIR}/${exe}' && cd '${TEST_OUTPUT_DIR}/${exe}' && \
               '$<TARGET_FILE:${exe}>' > '${GOLD_IMD_NAME}' && \
               diff -u '${GOLD_FILE_IMD}' '${GOLD_IMD_NAME}'"
        )
        set_tests_properties(${exe} PROPERTIES LABELS "kats")
    endforeach()

endforeach()
