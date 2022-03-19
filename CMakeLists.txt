cmake_minimum_required(VERSION 3.21)
project(DigitsRecognizer)
set(CMAKE_CXX_STANDARD 14)
add_executable(DigitsRecognizer ../src/main.cpp ../src/NeuralNetwork.hpp ../src/NeuralNetwork.cpp)
add_definitions(-Werror)
add_definitions(-O3)


#ImageMagick
add_definitions(-DMAGICKCORE_QUANTUM_DEPTH=16)
add_definitions(-DMAGICKCORE_HDRI_ENABLE=0)
find_package(ImageMagick)
find_package(ImageMagick COMPONENTS Magick++)
find_package(ImageMagick COMPONENTS convert)
find_package(ImageMagick COMPONENTS Magick++ convert)
include_directories(${ImageMagick_INCLUDE_DIRS})
target_link_libraries(DigitsRecognizer ${ImageMagick_LIBRARIES})