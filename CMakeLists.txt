cmake_minimum_required(VERSION 3.0)

project(Realidad-Aumentada)

find_package( OpenCV REQUIRED )
find_package( OpenGL REQUIRED )

add_subdirectory(aruco-1.3.0)
include_directories(aruco-1.3.0/include)

include_directories(${OpenCV_INCLUDE_DIR})
include_directories(${OpenGL_INCLUDE_DIR})

add_executable(Realidad-Aumentada ejemplo.cpp )

file(COPY intrinsics.yml video.avi DESTINATION data)

target_link_libraries(Realidad-Aumentada aruco ${OpenCV_LIBRARIES} ${OPENGL_LIBRARIES})
