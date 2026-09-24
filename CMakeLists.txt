cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(Axobrier
    VERSION   0.2.0
    LANGUAGES C CXX CUDA
    DESCRIPTION "Axobrier  -  System 1 Decision Engine"
)

# ─── Language Standards ───────────────────────────────────────────────
set(CMAKE_C_STANDARD          99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS        OFF)
set(CMAKE_CXX_STANDARD        17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS      OFF)
set(CMAKE_CUDA_STANDARD       17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ─── CUDA Architecture Targets ───────────────────────────────────────
# sm_89  = Ada Lovelace  (RTX 4090 / 4080 / 4070)
# sm_100 = Blackwell B100 / B200
# sm_120 = Blackwell consumer (RTX 5090 / 5080 / 5070)  -  compute 12.0
set(CMAKE_CUDA_ARCHITECTURES "89;100;120")

# ─── Compiler Flags ──────────────────────────────────────────────────
# C flags
if(MSVC)
    set(CMAKE_C_FLAGS_RELEASE   "/O2 /GL /DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /GL /DNDEBUG")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "/LTCG")
else()
    set(CMAKE_C_FLAGS_RELEASE   "-O3 -DNDEBUG -flto")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -flto")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "-flto")
endif()

# CUDA flags (applied to all configs, then per-config)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math --expt-relaxed-constexpr")
if(NOT MSVC)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -O3")
endif()

# ─── Sources ─────────────────────────────────────────────────────────
set(AXO_SOURCES
    src/engine.c
    src/memory_arena.cpp
    src/kernel_decision.cu
    src/kernel_encoder.cu
)

set(AXO_HEADERS
    include/axobrier_core.h
)

# ─── Shared Library Target ──────────────────────────────────────────
add_library(axobrier_core SHARED ${AXO_SOURCES})

target_include_directories(axobrier_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Link CUDA runtime (static to avoid runtime DLL dependency)
set_target_properties(axobrier_core PROPERTIES
    CUDA_RUNTIME_LIBRARY Static
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
    VERSION   ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    # Windows: export all C symbols marked AXO_API
    WINDOWS_EXPORT_ALL_SYMBOLS OFF
)

# ─── Platform Definitions ───────────────────────────────────────────
if(WIN32)
    target_compile_definitions(axobrier_core PRIVATE AXO_BUILD_DLL)
endif()

# ─── CUDA Toolkit (for cudaMemcpy, cudaStream, etc.) ────────────────
find_package(CUDAToolkit 12.0 REQUIRED)
target_link_libraries(axobrier_core PRIVATE CUDA::cudart_static)

# ─── Install Rules ──────────────────────────────────────────────────
include(GNUInstallDirs)
install(TARGETS axobrier_core
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}   # .dll on Windows
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(FILES ${AXO_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# ─── Test Harness ───────────────────────────────────────────────────
enable_testing()

add_executable(test_routing tests/test_routing.cpp)
target_link_libraries(test_routing PRIVATE axobrier_core CUDA::cudart_static)
target_include_directories(test_routing PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
set_target_properties(test_routing PROPERTIES
    CUDA_RUNTIME_LIBRARY Static
)

add_test(NAME routing_correctness COMMAND test_routing --correctness)
add_test(NAME routing_benchmark   COMMAND test_routing --benchmark)

add_executable(test_end_to_end tests/test_end_to_end.cpp)
target_link_libraries(test_end_to_end PRIVATE axobrier_core CUDA::cudart_static)
target_include_directories(test_end_to_end PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
set_target_properties(test_end_to_end PROPERTIES
    CUDA_RUNTIME_LIBRARY Static
)

add_test(NAME e2e_correctness COMMAND test_end_to_end --correctness)
add_test(NAME e2e_benchmark   COMMAND test_end_to_end --benchmark)

add_executable(test_cartridge tests/test_cartridge.cpp)
target_link_libraries(test_cartridge PRIVATE axobrier_core CUDA::cudart_static)
target_include_directories(test_cartridge PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
set_target_properties(test_cartridge PROPERTIES
    CUDA_RUNTIME_LIBRARY Static
)

add_test(NAME cartridge_hotswap COMMAND test_cartridge)
