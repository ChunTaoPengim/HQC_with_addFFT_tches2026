cmake_minimum_required(VERSION 3.21)
project(hqc VERSION 1.0 LANGUAGES C CXX)

# Enable testing & export compile commands
enable_testing()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -----------------------------------------------------------------------------
# ARCHITECTURE SELECTION
# -----------------------------------------------------------------------------
set(HQC_ARCH "ref" CACHE STRING "Choose implementation: ref or x86_64")
set_property(CACHE HQC_ARCH PROPERTY STRINGS ref x86_64)
if(NOT HQC_ARCH STREQUAL "ref" AND NOT HQC_ARCH STREQUAL "x86_64")
    message(FATAL_ERROR "HQC_ARCH must be 'ref' or 'x86_64'")
endif()

# -----------------------------------------------------------------------------
# COMPILE-TIME MACROS
# -----------------------------------------------------------------------------
if(HQC_ARCH STREQUAL "ref")
    add_compile_definitions(HQC_ARCH_REF=1)
else()
    add_compile_definitions(HQC_ARCH_X86_64=1)
    add_compile_definitions(HQC_ARCH_X86_64_AVX256=1)
endif()

# -----------------------------------------------------------------------------
# WARNING & OPTIMIZATION FLAGS (always on)
# -----------------------------------------------------------------------------
set(HQC_C_FLAGS
        -Wall -Werror -Wextra -Wmissing-prototypes -Wpedantic -Wredundant-decls
)

# -----------------------------------------------------------------------------
# ARCH-SPECIFIC OPTIMIZATIONS
# -----------------------------------------------------------------------------
if(HQC_ARCH STREQUAL "x86_64")
    if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
        list(APPEND HQC_C_FLAGS
                -funroll-loops
                -mavx -mavx2 -mbmi -mpclmul
                -Wno-error=ignored-optimization-argument
        )
    else()
        list(APPEND HQC_C_FLAGS
                -funroll-all-loops -flto -mavx -mavx2 -mbmi -mpclmul
        )
    endif()
endif()

# -----------------------------------------------------------------------------
# C STANDARD & SANITIZERS
# -----------------------------------------------------------------------------
set(HQC_C_STANDARD "99" CACHE STRING "Select a C standard")
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD ${HQC_C_STANDARD})

# Default build-type based optimization
if(CMAKE_BUILD_TYPE MATCHES Debug)
    list(APPEND HQC_C_FLAGS -O0)
else()
    list(APPEND HQC_C_FLAGS -O3)
endif()

set(HQC_SANITIZER "NONE" CACHE STRING "Select a sanitizer: NONE, ASAN, LSAN, MSAN, UBSAN")
mark_as_advanced(HQC_SANITIZER)
option(CLANG_FORMAT_ARCH "" "If non-empty, only format files under src/<this>; e.g. ref or x86_64")

if(HQC_SANITIZER STREQUAL "ASAN")
    list(REMOVE_ITEM HQC_C_FLAGS -O3 -O0)
    list(APPEND HQC_C_FLAGS -O1 -g -fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
elseif(HQC_SANITIZER STREQUAL "LSAN")
    list(REMOVE_ITEM HQC_C_FLAGS -O3 -O0)
    list(APPEND HQC_C_FLAGS -O1 -g -fsanitize=leak -fno-omit-frame-pointer)
    add_link_options(-fsanitize=leak)
elseif(HQC_SANITIZER STREQUAL "MSAN")
    list(REMOVE_ITEM HQC_C_FLAGS -O3 -O0)
    list(APPEND HQC_C_FLAGS -O1 -g -fsanitize=memory -fsanitize-memory-track-origins=2)
    add_link_options(-fsanitize=memory)
elseif(HQC_SANITIZER STREQUAL "UBSAN")
    list(REMOVE_ITEM HQC_C_FLAGS -O3 -O0)
    list(APPEND HQC_C_FLAGS -O1 -g -fsanitize=undefined)
    add_link_options(-fsanitize=undefined)
endif()

add_compile_options(${HQC_C_FLAGS})

# -----------------------------------------------------------------------------
# BUILD SUBDIRECTORIES
# -----------------------------------------------------------------------------
set(VARIANTS hqc-1 hqc-3 hqc-5)

add_subdirectory(lib)
add_subdirectory(src)
add_subdirectory(tests)

# -----------------------------------------------------------------------------
# FORMAT, STATIC ANALYSIS, PACKAGING
# -----------------------------------------------------------------------------
find_program(CLANG_FORMAT_EXE NAMES clang-format-18 clang-format)
if(CLANG_FORMAT_EXE)
    file(GLOB_RECURSE ALL_SRC
            src/*.c src/*.h
            lib/*.c lib/*.h
            tests/*.c tests/*.h tests/*.cpp
    )
    list(FILTER ALL_SRC EXCLUDE REGEX ".*/parameters\\.h$")
    if(CLANG_FORMAT_ARCH)
        list(FILTER ALL_SRC INCLUDE REGEX ".*/src/${CLANG_FORMAT_ARCH}/.*")
    endif()
    add_custom_target(check-format
            COMMAND ${CLANG_FORMAT_EXE} --dry-run --Werror --style=file ${ALL_SRC}
            COMMENT "Checking formatting with ${CLANG_FORMAT_EXE}"
    )
    add_custom_target(clang-format
            COMMAND ${CLANG_FORMAT_EXE} -i --style=file ${ALL_SRC}
            COMMENT "Reformatting all sources with ${CLANG_FORMAT_EXE}"
    )
endif()

find_program(CPPCHECK_EXE NAMES cppcheck)
if(CPPCHECK_EXE)
    file(GLOB_RECURSE CPPCHECK_SRCS
            src/*.c src/*.h
            lib/*.c lib/*.h
            tests/*.c tests/*.h tests/*.cpp
    )
    add_custom_target(cppcheck
            COMMAND ${CPPCHECK_EXE}
            --project=${CMAKE_BINARY_DIR}/compile_commands.json
            --enable=all
            --inconclusive
            --suppress=missingIncludeSystem
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT "Running cppcheck via compile_commands.json"
            VERBATIM
    )
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/packaging/build_nist_release.sh")
    add_custom_target(package-release
            COMMAND bash "${CMAKE_SOURCE_DIR}/packaging/build_nist_release.sh"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            COMMENT "Assembling NIST Release under packaging/Release"
    )
endif()
