#pragma once

#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <vector>

struct Detection {
    cv::Rect bbox;
    float confidence;
    int class_id;
    std::string label;
};

class InferenceManager {
public:
    InferenceManager();
    ~InferenceManager();
    
    bool loadModel(const std::string& model_path);
    std::vector<Detection> processFrame(const cv::Mat& frame);
    void setDebug(bool debug) { this->debug = debug; }
    
private:
    bool debug = false;
    float conf_threshold = 0.5f;  // Confidence threshold for faces
    float nms_threshold = 0.4f;   // NMS threshold
    
    torch::jit::script::Module model;
    cv::Size input_size;
    
    torch::Tensor preprocess(const cv::Mat& frame);
    std::vector<Detection> postprocess(torch::Tensor& output, const cv::Size& frame_size);
    void applyNMS(std::vector<Detection>& detections);
};
