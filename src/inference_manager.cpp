#include "inference_manager.h"
#include <iostream>

InferenceManager::InferenceManager() {
    input_size = cv::Size(640, 640);
    conf_threshold = 0.5f;
    nms_threshold = 0.4f;
}

InferenceManager::~InferenceManager() {}

bool InferenceManager::loadModel(const std::string& model_path) {
    try {
        model = torch::jit::load(model_path);
        model.eval();
        model.to(torch::kCPU);
        
        std::cout << "Face detection model loaded successfully" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return false;
    }
}

torch::Tensor InferenceManager::preprocess(const cv::Mat& frame) {
    cv::Mat resized;
    cv::resize(frame, resized, input_size);
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    
    torch::Tensor tensor = torch::from_blob(
        resized.data, {input_size.height, input_size.width, 3}, torch::kByte);
    
    tensor = tensor.permute({2, 0, 1})
                  .to(torch::kFloat32)
                  .div_(255.0)
                  .unsqueeze(0);
    
    return tensor;
}

void InferenceManager::applyNMS(std::vector<Detection>& detections) {
    if (detections.empty()) return;
    
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    
    for (const auto& det : detections) {
        boxes.push_back(det.bbox);
        scores.push_back(det.confidence);
    }
    
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, conf_threshold, nms_threshold, indices);
    
    std::vector<Detection> filtered_detections;
    for (int idx : indices) {
        filtered_detections.push_back(detections[idx]);
    }
    
    detections = std::move(filtered_detections);
}

std::vector<Detection> InferenceManager::postprocess(torch::Tensor& output, const cv::Size& frame_size) {
    std::vector<Detection> results;
    
    try {
        if (!output.defined() || output.numel() == 0) {
            return results;
        }

        if (debug) {
            std::cout << "Raw output shape: [";
            for (int64_t dim : output.sizes()) {
                std::cout << dim << " ";
            }
            std::cout << "]" << std::endl;
        }

        // Face detection model output: [batch_size, num_predictions, 5]
        // 5 values: [x_center, y_center, width, height, confidence]
        if (output.dim() != 3 || output.size(2) != 5) {
            std::cerr << "Expected [batch, predictions, 5] tensor, got shape: ";
            for (int64_t dim : output.sizes()) {
                std::cerr << dim << " ";
            }
            std::cerr << std::endl;
            return results;
        }
        
        int batch_size = output.size(0);
        int num_predictions = output.size(1);
        
        if (debug) {
            std::cout << "Processing " << num_predictions << " face predictions" << std::endl;
        }
        
        auto accessor = output.accessor<float,3>();
        
        for (int i = 0; i < num_predictions; ++i) {
            // Lấy confidence (index 4)
            float conf = accessor[0][i][4];
            if (conf < conf_threshold) continue;
            
            // Lấy tọa độ normalized [0,1]
            float x_center = accessor[0][i][0];
            float y_center = accessor[0][i][1];
            float width = accessor[0][i][2];
            float height = accessor[0][i][3];
            
            // Kiểm tra tọa độ hợp lệ
            if (x_center < 0 || x_center > 1 || y_center < 0 || y_center > 1 ||
                width <= 0 || width > 1 || height <= 0 || height > 1) {
                if (debug) {
                    std::cout << "Invalid normalized coords: x=" << x_center << ", y=" << y_center 
                              << ", w=" << width << ", h=" << height << std::endl;
                }
                continue;
            }
            
            // Tính toán face bounding box
            float x1_norm = x_center - width / 2.0f;
            float y1_norm = y_center - height / 2.0f;
            float x2_norm = x_center + width / 2.0f;
            float y2_norm = y_center + height / 2.0f;
            
            // Clamp về [0,1]
            x1_norm = std::max(0.0f, std::min(1.0f, x1_norm));
            y1_norm = std::max(0.0f, std::min(1.0f, y1_norm));
            x2_norm = std::max(0.0f, std::min(1.0f, x2_norm));
            y2_norm = std::max(0.0f, std::min(1.0f, y2_norm));
            
            // Chuyển sang pixel coordinates
            int x1 = (int)(x1_norm * frame_size.width);
            int y1 = (int)(y1_norm * frame_size.height);
            int x2 = (int)(x2_norm * frame_size.width);
            int y2 = (int)(y2_norm * frame_size.height);
            
            // Tính width và height
            int bbox_width = x2 - x1;
            int bbox_height = y2 - y1;
            
            // Kiểm tra kích thước face hợp lý (không quá nhỏ, không quá lớn)
            int min_face_size = std::min(frame_size.width, frame_size.height) * 0.03; // 3% của frame
            int max_face_size = std::min(frame_size.width, frame_size.height) * 0.8;  // 80% của frame
            
            if (bbox_width < min_face_size || bbox_height < min_face_size ||
                bbox_width > max_face_size || bbox_height > max_face_size) {
                if (debug) {
                    std::cout << "Face size out of range: w=" << bbox_width << ", h=" << bbox_height 
                              << " (min=" << min_face_size << ", max=" << max_face_size << ")" << std::endl;
                }
                continue;
            }
            
            // Tạo face detection
            Detection det;
            det.bbox = cv::Rect(x1, y1, bbox_width, bbox_height);
            det.confidence = conf;
            det.class_id = 0;  // Only one class: face
            det.label = "face";
            
            results.push_back(det);
            
            if (debug) {
                std::cout << "Face detection: conf=" << det.confidence 
                          << " bbox=[" << det.bbox.x << "," << det.bbox.y << "," 
                          << det.bbox.width << "," << det.bbox.height << "]" << std::endl;
            }
        }
        
        if (debug) {
            std::cout << "Before NMS: " << results.size() << " face detections" << std::endl;
        }
        
        applyNMS(results);
        
        if (debug) {
            std::cout << "After NMS: " << results.size() << " face detections" << std::endl;
        }
        
    } 
    catch (const std::exception& e) {
        std::cerr << "Face detection postprocessing exception: " << e.what() << std::endl;
    }
    
    return results;
}

std::vector<Detection> InferenceManager::processFrame(const cv::Mat& frame) {
    auto input_tensor = preprocess(frame);
    input_tensor = input_tensor.to(torch::kCPU);
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);
    
    torch::Tensor output;
    {
        torch::NoGradGuard no_grad;
        output = model.forward(inputs).toTensor();
    }
    
    return postprocess(output, frame.size());
}
