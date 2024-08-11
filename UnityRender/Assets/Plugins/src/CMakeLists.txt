cmake_minimum_required(VERSION 3.8)
project("EigenForUnity")
# message(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/../../../../ThirdParty/eigen-3.3.8")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/../../../../ThirdParty/EigenExtension")
AUX_SOURCE_DIRECTORY("${CMAKE_CURRENT_SOURCE_DIR}/../../../../ThirdParty/EigenExtension" EigenExtensionDir)

add_library(${PROJECT_NAME} SHARED Eigen.cpp Network.cpp ${EigenExtensionDir})
# add_executable(${PROJECT_NAME} Main.cpp Eigen.cpp Network.cpp ${EigenExtensionDir})