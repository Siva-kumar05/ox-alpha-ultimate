cmake_minimum_required(VERSION 3.20)
project(OxAlphaExecutionEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native -flto")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -funroll-loops")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fomit-frame-pointer -fno-exceptions -fno-rtti")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter")
elseif(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /arch:AVX2 /fp:fast /GL /LTCG")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Oi /Ot /Ob2 /GF /Gy")
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Execution engine library
add_library(execution_engine STATIC
    src/execution_engine.cpp
)

target_include_directories(execution_engine PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Market data decoder library
add_library(market_data_decoder STATIC
    src/market_data_decoder.cpp
)

target_include_directories(market_data_decoder PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Main execution engine executable
add_executable(ox_execution_engine
    src/main.cpp
)

target_link_libraries(ox_execution_engine
    execution_engine
    market_data_decoder
    pthread
    rt
)

# Tests
enable_testing()
add_subdirectory(tests)

# Install
install(TARGETS execution_engine market_data_decoder ox_execution_engine
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)