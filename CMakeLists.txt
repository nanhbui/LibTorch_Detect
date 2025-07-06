cmake_minimum_required(VERSION 3.16)
project(FaceDetectPlugin)

# Thiết lập đường dẫn LibTorch - QUAN TRỌNG
set(LIBTORCH_DIR "${CMAKE_SOURCE_DIR}/libtorch")  # Giả sử libtorch nằm cùng thư mục project
set(Torch_DIR "${LIBTORCH_DIR}/share/cmake/Torch")

# Thiết lập các tùy chọn cơ bản
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Tìm kiếm LibTorch
find_package(Torch REQUIRED PATHS ${LIBTORCH_DIR} NO_DEFAULT_PATH)
if(NOT Torch_FOUND)
    message(FATAL_ERROR "LibTorch not found at ${LIBTORCH_DIR}")
else()
    message(STATUS "Found LibTorch: ${TORCH_INCLUDE_DIRS}")
endif()

# Tìm kiếm OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found, install with: sudo apt install libopencv-dev")
endif()

# Tìm kiếm GStreamer
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0>=1.16)
pkg_check_modules(GSTREAMER_BASE REQUIRED gstreamer-base-1.0)
pkg_check_modules(GSTREAMER_VIDEO REQUIRED gstreamer-video-1.0)

# Thiết lập thư mục include
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
    ${GSTREAMER_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# Tạo plugin GStreamer
add_library(gstfacedetect MODULE
    src/gst_face_detect.cpp
    src/inference_manager.cpp
)


# Trong phần target_link_libraries cho gstfacedetect
target_link_libraries(gstfacedetect PRIVATE
    ${TORCH_LIBRARIES}
    ${GSTREAMER_LIBRARIES}
    ${GSTREAMER_BASE_LIBRARIES}
    ${GSTREAMER_VIDEO_LIBRARIES}  # THÊM DÒNG NÀY
    ${OpenCV_LIBS}
)

# Cấu hình đặc biệt cho plugin
set_target_properties(gstfacedetect PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    CXX_STANDARD_REQUIRED YES
)

# Tạo ứng dụng test (nếu cần)
add_executable(face_detect_test src/main.cpp)
target_link_libraries(face_detect_test PRIVATE
    ${GSTREAMER_LIBRARIES}
    ${OpenCV_LIBS}
)

# Cài đặt plugin
install(TARGETS gstfacedetect
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/gstreamer-1.0
)
