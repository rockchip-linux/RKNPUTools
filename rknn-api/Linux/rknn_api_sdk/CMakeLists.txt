cmake_minimum_required(VERSION 2.8)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


include_directories(
    ${CMAKE_SOURCE_DIR}/3rdparty/opencv/include
    ${CMAKE_SOURCE_DIR}/rknn_api/include
)

set(link_libs  ${CMAKE_SOURCE_DIR}/3rdparty/opencv/lib64/libopencv_core.so
               ${CMAKE_SOURCE_DIR}/3rdparty/opencv/lib64/libopencv_highgui.so
               ${CMAKE_SOURCE_DIR}/3rdparty/opencv/lib64/libopencv_imgcodecs.so
               ${CMAKE_SOURCE_DIR}/3rdparty/opencv/lib64/libopencv_imgproc.so
               ${CMAKE_SOURCE_DIR}/rknn_api/lib64/librknn_api.so
               pthread
)

add_executable(rknn_mobilenet  rknn_mobilenet.cpp)
target_link_libraries(rknn_mobilenet ${link_libs})

add_executable(rknn_ssd  rknn_ssd.cpp)
target_link_libraries(rknn_ssd ${link_libs})
