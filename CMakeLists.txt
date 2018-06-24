cmake_minimum_required(VERSION 3.9)
project(CalibrationCPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

find_package(OpenCV REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")

set(SOURCE_FILES
        findboxcorners.h findboxcorners.cpp
        findboxplanes.h findboxplanes.cpp
        utils.h utils.cpp
        boxsize.hpp
        main.cpp plane.hpp
        select_image_corners.h select_image_corners.cpp)

add_executable(CalibrationCPP ${SOURCE_FILES})

target_link_libraries(CalibrationCPP ${OpenCV_LIBS})
