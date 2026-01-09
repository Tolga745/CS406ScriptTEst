cmake_minimum_required(VERSION 3.9)
project(ConTree)
project(ConTree LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

find_package(OpenMP REQUIRED)

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)

# Set default build type to Release if not specified
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    message(STATUS "Build type not specified: Use Release by default")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Compiling in release mode")
else()
    message(STATUS "Compiling in debug mode")    
    add_compile_definitions(DEBUG=1)
endif()

if (CMAKE_BUILD_TYPE STREQUAL Release)
    
    if (MSVC)
        message(STATUS "Using MSVC compiler")
        add_compile_options(/Ot)
    else()
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-O3>)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-funroll-loops>)
    endif()

endif()
   
    

# Gather source files
file(GLOB SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/DataStructures/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Utilities/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Engine/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DataStructures/src/dataset.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/Engine/src/gpu_solver.cu
)

# Define include directories
set(INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/Utilities/include
    ${CMAKE_SOURCE_DIR}/DataStructures/include
    ${CMAKE_SOURCE_DIR}/Engine/include
)

# Specify include directories
include_directories(${INCLUDE_DIRS})

# Create the executable
add_executable(ConTree ${SRC_FILES})

target_compile_options(ConTree PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_link_libraries(ConTree PUBLIC OpenMP::OpenMP_CXX)

# Set target properties based on build type
if (CMAKE_BUILD_TYPE STREQUAL Release)        
    if( ipo_supported )
	    message(STATUS "IPO / LTO enabled")    
        set_property(TARGET ConTree PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
         message(STATUS "IPO / LTO not supported: <${ipo_error}>")
    endif()
else() 
    
    # enable runtime profilling (with MSVC)
    if (MSVC)
        message(STATUS "Enabling profiler")    
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /PROFILE")
    else ()
        message(STATUS "No profiler")
    endif()
endif()

