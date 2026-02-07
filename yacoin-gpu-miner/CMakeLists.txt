cmake_minimum_required(VERSION 3.18)
project(scrypt-miner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(WITH_CUDA "Build with CUDA support (NVIDIA)" ON)
option(WITH_OPENCL "Build with OpenCL support (AMD)" OFF)

# Find packages
find_package(Threads REQUIRED)

# JSON library (optional)
find_package(PkgConfig)
if(PkgConfig_FOUND)
    pkg_check_modules(JSON json-c)
endif()

# Source files
set(CORE_SOURCES
    src/main.cpp
    src/core/miner.cpp
    src/core/stratum.cpp
    src/core/util.cpp
)

set(CORE_HEADERS
    src/core/miner.h
    src/core/stratum.h
)

# CUDA support (NVIDIA)
if(WITH_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    set(CUDA_SOURCES
        src/cuda/adaptivepow.cu
    )

    add_library(cuda_miner STATIC ${CUDA_SOURCES})
    target_include_directories(cuda_miner PRIVATE src)
    set_target_properties(cuda_miner PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        # Support: Pascal (10xx), Volta (Titan V), Turing (20xx), Ampere (30xx), Ada (40xx)
        CUDA_ARCHITECTURES "61;70;75;80;86;89;90"
    )

    add_definitions(-DWITH_CUDA)
endif()

# OpenCL support (AMD)
if(WITH_OPENCL)
    find_package(OpenCL REQUIRED)

    set(OPENCL_SOURCES
        src/opencl/opencl_miner.cpp
    )

    add_definitions(-DWITH_OPENCL)
endif()

# Main executable
add_executable(scrypt-miner ${CORE_SOURCES} ${CORE_HEADERS})

target_include_directories(scrypt-miner PRIVATE src)

# Link libraries
target_link_libraries(scrypt-miner PRIVATE Threads::Threads)

if(WITH_CUDA)
    target_link_libraries(scrypt-miner PRIVATE cuda_miner CUDA::cudart)
endif()

if(WITH_OPENCL)
    target_link_libraries(scrypt-miner PRIVATE OpenCL::OpenCL)
endif()

if(JSON_FOUND)
    target_include_directories(scrypt-miner PRIVATE ${JSON_INCLUDE_DIRS})
    target_link_libraries(scrypt-miner PRIVATE ${JSON_LIBRARIES})
endif()

# Windows specific
if(WIN32)
    target_link_libraries(scrypt-miner PRIVATE ws2_32)
endif()

# macOS/Linux specific
if(UNIX AND NOT APPLE)
    find_package(OpenSSL)
    if(OpenSSL_FOUND)
        target_link_libraries(scrypt-miner PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    endif()
endif()

# Install targets
install(TARGETS scrypt-miner RUNTIME DESTINATION bin)

# Copy OpenCL kernels for runtime loading
if(WITH_OPENCL)
    install(FILES
        src/opencl/adaptivepow.cl
        DESTINATION share/scrypt-miner
    )
endif()

# Print configuration
message(STATUS "")
message(STATUS "===========================================")
message(STATUS "  Scrypt Coin GPU Miner Configuration")
message(STATUS "===========================================")
message(STATUS "  CUDA support (NVIDIA): ${WITH_CUDA}")
message(STATUS "  OpenCL support (AMD):  ${WITH_OPENCL}")
message(STATUS "===========================================")
message(STATUS "")
