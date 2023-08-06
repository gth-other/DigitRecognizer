cmake_minimum_required(VERSION 3.25)
project(DigitRecognizer)
set(CMAKE_CXX_STANDARD 20)
add_definitions(-Werror)
add_definitions(-O3)
add_definitions(-flto)
add_executable(DigitRecognizer src/main.cpp src/Tensor2D.hpp src/Tensor2D.cpp src/Perceptron.hpp src/Perceptron.cpp)


target_link_libraries(DigitRecognizer sfml-system sfml-window sfml-graphics)