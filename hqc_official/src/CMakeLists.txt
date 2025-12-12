cmake_minimum_required(VERSION 3.21)
project(hqc-src C)

# List of HQC parameter variants
set(VARIANTS "hqc-1;hqc-3;hqc-5" CACHE INTERNAL "List of variants")

# 1) cross-arch common sources
file(GLOB COMMON_SOURCES
        "${CMAKE_SOURCE_DIR}/src/common/*.c"
)

# 2) arch-common: shared across variants (top-level .c)
if(HQC_ARCH STREQUAL "ref")
    set(ARCH_COMMON_DIR "${CMAKE_SOURCE_DIR}/src/ref")
else()
    set(ARCH_COMMON_DIR "${CMAKE_SOURCE_DIR}/src/x86_64/common")
endif()
file(GLOB ARCH_COMMON_TOP_SOURCES
        "${ARCH_COMMON_DIR}/*.c"
)

# 3) arch-specific implementation sources
if(HQC_ARCH STREQUAL "ref")
    set(PLATFORM_DIR "${CMAKE_SOURCE_DIR}/src/ref")
else()
    set(PLATFORM_DIR "${CMAKE_SOURCE_DIR}/src/x86_64/${HQC_X86_IMPL}")
endif()
file(GLOB PLATFORM_SOURCES
        "${PLATFORM_DIR}/*.c"
)

# 4) build one static library per variant
foreach(var IN LISTS VARIANTS)
    string(REPLACE "-" "_" safe ${var})

    if(HQC_ARCH STREQUAL "ref")
        set(VAR_DIR "${CMAKE_SOURCE_DIR}/src/ref/${var}")
    else()
        set(VAR_DIR "${CMAKE_SOURCE_DIR}/src/x86_64/${HQC_X86_IMPL}/${var}")
    endif()
    file(GLOB VARIANT_SOURCES
            "${VAR_DIR}/*.c"
    )

    file(GLOB ARCH_COMMON_VARIANT_SOURCES
            "${ARCH_COMMON_DIR}/${var}/*.c"
    )

    add_library(${safe}_${HQC_ARCH} STATIC
            ${COMMON_SOURCES}
            ${ARCH_COMMON_TOP_SOURCES}
            ${ARCH_COMMON_VARIANT_SOURCES}
            ${PLATFORM_SOURCES}
            ${VARIANT_SOURCES}
    )

    target_include_directories(${safe}_${HQC_ARCH} PUBLIC
            "${CMAKE_SOURCE_DIR}/src/common"
            "${CMAKE_SOURCE_DIR}/src/common/${var}"
            "${ARCH_COMMON_DIR}"
            "${ARCH_COMMON_DIR}/${var}"
            "${PLATFORM_DIR}"
            "${VAR_DIR}"
            "${CMAKE_SOURCE_DIR}/lib/fips202"
    )

    target_link_libraries(${safe}_${HQC_ARCH} PUBLIC
            fips202
    )
endforeach()
