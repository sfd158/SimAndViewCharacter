PROJECT(TestWm5)
 
cmake_minimum_required(VERSION 3.2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_definitions(_USE_MATH_DEFINES)

if (WIN32)
add_compile_definitions(_CRT_SECURE_NO_DEPRECATE)
else()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

set(dir ${CMAKE_CURRENT_SOURCE_DIR})
set(EXECUTABLE_OUTPUT_PATH ${dir} CACHE PATH "Build directory" FORCE)
set(LIBRARY_OUTPUT_PATH ${dir} CACHE PATH "Build directory" FORCE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${dir})
set(CMAKE_BUILD_FILES_DIRECTORY ${dir})
set(CMAKE_BUILD_DIRECTORY ${dir})
set(CMAKE_BINARY_DIR  ${dir})
SET(EXECUTABLE_OUTPUT_PATH ${dir})
SET(LIBRARY_OUTPUT_PATH ${dir})
SET(CMAKE_CACHEFILE_DIR ${dir})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../ThirdParty/eigen-3.3.8/Eigen)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../ThirdParty/eigen-3.3.8)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../ThirdParty)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Wm5)
# aux_source_directory("Wm5" DIR_WM5)

add_executable(${PROJECT_NAME} main.cpp Wm5/Wm5Math.cpp Wm5/Wm5Assert.cpp) 
