# 🔍 Face Detection GStreamer Plugin (LibTorch + OpenCV + GStreamer)

Dự án này xây dựng một **plugin GStreamer tùy chỉnh** dùng LibTorch để **nhận diện khuôn mặt theo thời gian thực** từ camera (hoặc video). Tích hợp mô hình đã được huấn luyện (TorchScript), và có thể hiển thị bounding box bằng OpenCV.

## 📁 Cấu trúc thư mục

```
LibTorch_Proj/
├── libtorch/                    # Thư viện LibTorch (giải nén từ bản tải về)
│   ├── include/
│   ├── lib/
│   └── share/
├── CMakeLists.txt              # File cấu hình build
├── export_model.py             # Script Python để export mô hình .pt
├── include/
│   ├── gst_face_detect.h
│   └── inference_manager.h
└── src/
    ├── gst_face_detect.cpp
    ├── inference_manager.cpp
    └── main.cpp                # (tuỳ chọn) file kiểm tra GStreamer pipeline bằng C++
```

## ⚙️ Hướng dẫn cài đặt và chạy

### 1. Clone và tải LibTorch (CPU version)

```bash
git clone https://github.com/<your-name>/LibTorch_Proj.git
cd LibTorch_Proj

# Tải LibTorch CPU về và giải nén
wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.1.2%2Bcpu.zip
unzip libtorch-shared-with-deps-2.1.2+cpu.zip
```

> **Lưu ý:** Thư mục `libtorch/` cần nằm trong `LibTorch_Proj/`

### 2. Tạo môi trường ảo và export mô hình

```bash
# Cài Python virtualenv (nếu chưa có)
sudo apt install python3-venv

# Tạo môi trường và kích hoạt
python3 -m venv venv
source venv/bin/activate

# Cài đặt thư viện cần thiết
pip install torch torchvision

# Chạy script export mô hình
python3 export_model.py
```

### 3. Build plugin bằng CMake

```bash
# Tạo thư mục build và chuyển vào
mkdir build && cd build

# Gọi CMake và chỉ rõ LibTorch
cmake ..
make -j$(nproc)
```

### 4. Thiết lập biến môi trường và kiểm tra plugin

```bash
export GST_PLUGIN_PATH=$PWD
export LD_LIBRARY_PATH=../libtorch/lib:$LD_LIBRARY_PATH

gst-inspect-1.0 facedetect
```

### 5. Chạy pipeline kiểm tra với webcam

```bash
# Với mô hình compact
gst-launch-1.0 v4l2src device=/dev/video0 ! videoconvert ! facedetect model-path=/home/lenovo/DA1/LibTorch_Proj/face_detector_compact.pt debug=true ! videoconvert ! autovideosink

# Với mô hình mobile
gst-launch-1.0 v4l2src device=/dev/video0 ! videoconvert ! facedetect model-path=/home/lenovo/DA1/LibTorch_Proj/face_detector_mobile.pt debug=true ! videoconvert ! autovideosink
```

> **Lưu ý:** Thay đổi `model-path` nếu bạn đang dùng đường dẫn khác.

## 🛠️ Cấu hình trong `CMakeLists.txt`

Đã thiết lập sẵn để:
- Tự động tìm **LibTorch** tại `libtorch/`
- Link với GStreamer (`gstreamer-1.0`, `gstreamer-video-1.0`, v.v.)
- Sử dụng OpenCV để vẽ bounding box (`cv::rectangle`, `cv::putText`)
- Xuất `.so` plugin ra thư mục `build/` và có thể gọi từ GStreamer

## 📌 Yêu cầu hệ thống

- Ubuntu 20.04+ (đã test trên 22.04)
- CMake ≥ 3.16
- GStreamer ≥ 1.16 (cài bằng `sudo apt install libgstreamer1.0-dev`)
- OpenCV (cài bằng `sudo apt install libopencv-dev`)
- Python 3.8+ để export model

## 📷 Kết quả

Hiển thị video webcam với các bounding box được vẽ quanh khuôn mặt phát hiện được, sử dụng mô hình TorchScript.

## 📜 Giấy phép

MIT License © 2025
