# 设置cmake版本，如果cmake版本过高，可能会出现错误
cmake_minimum_required(VERSION 3.5.1)
project(NCNN_test)

# 设置C++编译版本
set(CMAKE_CXX_STANDARD 17)

# 添加编译选项
SET(CMAKE_CXX_FLAGS "-std=c++17 -O2")

set(PROJECT_SOURCE_DIR ../)
set(WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

# 生成文件的输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# 分别设置ncnn的链接库和头文件
set(NCNN_LIBS E:/environments/C++/ncnn/ncnn/install_msvc/lib/ncnn.lib)
set(NCNN_INCLUDE_DIRS E:/environments/C++/ncnn/ncnn/install_msvc/include)
include_directories(${NCNN_INCLUDE_DIRS})


# 配置OpenCV
set(OpenCV_DIR D:/environments/C++/OpenCV/opencv-msvc/build/x64/vc15/lib)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


# 建立链接依赖
add_executable(NCNN src/inference.cpp)
target_link_libraries(NCNN ${NCNN_LIBS})
target_link_libraries(NCNN ${OpenCV_LIBS})