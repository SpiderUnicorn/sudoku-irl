include_directories(${sudoku_SOURCE_DIR}/src/model)
# I assume you want to use LibProject as a library in MainProject.
include_directories(${MyProject_SOURCE_DIR}/LibProject/inc)
link_directories(${MyProject_SOURCE_DIR}/LibProject/src)
add_executable(MainProject main.cpp)
target_link_libraries(MainProject LibProject)