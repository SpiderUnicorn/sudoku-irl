cmake_minimum_required(VERSION 2.8)

set (CMAKE_CXX_STANDARD 11)

project( sudoku )

if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")
endif()

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "src/*.h")

# open cv
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories("src")


add_executable( sudoku ./src/main.cpp ./src/pre_process.cpp ./src/digit_recognizer.cpp )
target_link_libraries( sudoku ${OpenCV_LIBS} )
