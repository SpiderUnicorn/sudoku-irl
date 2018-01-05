cmake_minimum_required(VERSION 2.8)

set (CMAKE_CXX_STANDARD 17)

project( sudoku )

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "src/*.h")

# open cv
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( sudoku ${OpenCV_LIBS} )

include_directories("src")

add_executable( sudoku ./src/main.cpp ./src/pre_process.cpp ./src/digit_recognizer.cpp )


