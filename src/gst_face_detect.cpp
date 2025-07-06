#include "gst_face_detect.h"
#include "inference_manager.h"
#include <opencv2/opencv.hpp>
#define PACKAGE "facedetect"
#define VERSION "1.0"
#define GST_PACKAGE_NAME "Face Detection Plugin"
#define GST_PACKAGE_ORIGIN "http://example.com"
GST_DEBUG_CATEGORY_STATIC(gst_face_detect_debug);
#define GST_CAT_DEFAULT gst_face_detect_debug

enum {
    PROP_0,
    PROP_MODEL_PATH,
    PROP_PASSTHROUGH,
    PROP_DEBUG
};

static GstStaticPadTemplate gst_face_detect_sink_template =
    GST_STATIC_PAD_TEMPLATE("sink",
        GST_PAD_SINK,
        GST_PAD_ALWAYS,
        GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE("RGB")));

static GstStaticPadTemplate gst_face_detect_src_template =
    GST_STATIC_PAD_TEMPLATE("src",
        GST_PAD_SRC,
        GST_PAD_ALWAYS,
        GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE("RGB")));

static gboolean plugin_init(GstPlugin *plugin) {
    GST_DEBUG_CATEGORY_INIT(gst_face_detect_debug, "facedetect", 0, "Face Detection Plugin");
    return gst_element_register(plugin, "facedetect", GST_RANK_NONE, GST_TYPE_FACE_DETECT);
}

G_DEFINE_TYPE(GstFaceDetect, gst_face_detect, GST_TYPE_BASE_TRANSFORM)

static void gst_face_detect_set_property(GObject *object, guint prop_id,
                                       const GValue *value, GParamSpec *pspec);
static void gst_face_detect_get_property(GObject *object, guint prop_id,
                                       GValue *value, GParamSpec *pspec);
static GstFlowReturn gst_face_detect_transform_ip(GstBaseTransform *base,
                                                GstBuffer *buf);
static gboolean gst_face_detect_set_caps(GstBaseTransform *base,
                                       GstCaps *incaps, GstCaps *outcaps);
static gboolean gst_face_detect_start(GstBaseTransform *base);
static gboolean gst_face_detect_stop(GstBaseTransform *base);

static void gst_face_detect_class_init(GstFaceDetectClass *klass) {
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
    GstBaseTransformClass *base_transform_class = GST_BASE_TRANSFORM_CLASS(klass);
    GstElementClass *element_class = GST_ELEMENT_CLASS(klass);

    gst_element_class_set_static_metadata(element_class,
        "Face Detection Element",
        "Filter/Effect/Video",
        "Performs face detection using PyTorch",
        "Your Name <your.email@example.com>");

    gst_element_class_add_static_pad_template(element_class, &gst_face_detect_sink_template);
    gst_element_class_add_static_pad_template(element_class, &gst_face_detect_src_template);

    gobject_class->set_property = gst_face_detect_set_property;
    gobject_class->get_property = gst_face_detect_get_property;

    g_object_class_install_property(gobject_class, PROP_MODEL_PATH,
        g_param_spec_string("model-path", "Model Path",
                          "Path to TorchScript model",
                          NULL, G_PARAM_READWRITE));
    
    g_object_class_install_property(gobject_class, PROP_PASSTHROUGH,
        g_param_spec_boolean("passthrough", "Passthrough",
                           "Bypass processing when TRUE",
                           FALSE, G_PARAM_READWRITE));
    
    g_object_class_install_property(gobject_class, PROP_DEBUG,
        g_param_spec_boolean("debug", "Debug",
                           "Enable debug output",
                           FALSE, G_PARAM_READWRITE));

    base_transform_class->transform_ip = GST_DEBUG_FUNCPTR(gst_face_detect_transform_ip);
    base_transform_class->set_caps = GST_DEBUG_FUNCPTR(gst_face_detect_set_caps);
    base_transform_class->start = GST_DEBUG_FUNCPTR(gst_face_detect_start);
    base_transform_class->stop = GST_DEBUG_FUNCPTR(gst_face_detect_stop);
}

static void gst_face_detect_init(GstFaceDetect *filter) {
    filter->model_path = NULL;
    filter->passthrough = FALSE;
    filter->debug = FALSE;
    gst_video_info_init(&filter->video_info);
    filter->inference_manager = new InferenceManager();
}

static void gst_face_detect_set_property(GObject *object, guint prop_id,
                                       const GValue *value, GParamSpec *pspec) {
    GstFaceDetect *filter = GST_FACE_DETECT(object);
    
    switch (prop_id) {
        case PROP_MODEL_PATH:
            g_free(filter->model_path);
            filter->model_path = g_value_dup_string(value);
            break;
        case PROP_PASSTHROUGH:
            filter->passthrough = g_value_get_boolean(value);
            break;
        case PROP_DEBUG:
            filter->debug = g_value_get_boolean(value);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
    }
}

static void gst_face_detect_get_property(GObject *object, guint prop_id,
                                       GValue *value, GParamSpec *pspec) {
    GstFaceDetect *filter = GST_FACE_DETECT(object);
    
    switch (prop_id) {
        case PROP_MODEL_PATH:
            g_value_set_string(value, filter->model_path);
            break;
        case PROP_PASSTHROUGH:
            g_value_set_boolean(value, filter->passthrough);
            break;
        case PROP_DEBUG:
            g_value_set_boolean(value, filter->debug);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
    }
}

static gboolean gst_face_detect_set_caps(GstBaseTransform *base,
                                       GstCaps *incaps, GstCaps *outcaps) {
    GstFaceDetect *filter = GST_FACE_DETECT(base);
    
    if (!gst_video_info_from_caps(&filter->video_info, incaps)) {
        GST_ERROR_OBJECT(filter, "Failed to parse input caps");
        return FALSE;
    }

    return TRUE;
}

static gboolean gst_face_detect_start(GstBaseTransform *base) {
    GstFaceDetect *filter = GST_FACE_DETECT(base);
    
    if (filter->passthrough) {
        GST_INFO_OBJECT(filter, "Starting in passthrough mode");
        return TRUE;
    }

    if (!filter->model_path) {
        GST_ERROR_OBJECT(filter, "No model path specified");
        return FALSE;
    }

    InferenceManager *im = static_cast<InferenceManager*>(filter->inference_manager);
    if (!im->loadModel(filter->model_path)) {
        GST_ERROR_OBJECT(filter, "Failed to load model");
        return FALSE;
    }
    
    return TRUE;
}

static gboolean gst_face_detect_stop(GstBaseTransform *base) {
    GstFaceDetect *filter = GST_FACE_DETECT(base);
    InferenceManager *im = static_cast<InferenceManager*>(filter->inference_manager);
    delete im;
    filter->inference_manager = nullptr;
    return TRUE;
}

// Phần vẽ detection trong gst_face_detect_transform_ip
static GstFlowReturn gst_face_detect_transform_ip(GstBaseTransform* base, GstBuffer* buf) {
    GstFaceDetect* filter = GST_FACE_DETECT(base);
    
    if (filter->passthrough) return GST_FLOW_OK;

    GstMapInfo map;
    if (!gst_buffer_map(buf, &map, GST_MAP_READWRITE)) {
        GST_ERROR_OBJECT(filter, "Failed to map buffer");
        return GST_FLOW_ERROR;
    }

    try {
        int width = GST_VIDEO_INFO_WIDTH(&filter->video_info);
        int height = GST_VIDEO_INFO_HEIGHT(&filter->video_info);
        
        if (filter->debug) {
            GST_INFO_OBJECT(filter, "Processing frame: %dx%d", width, height);
        }
        
        // Tạo cv::Mat từ buffer - CHÚ Ý: GST format là RGB, không phải BGR
        cv::Mat frame(height, width, CV_8UC3, map.data);
        
        // Clone frame để không ảnh hưởng đến original buffer khi chuyển đổi màu
        cv::Mat frame_bgr = frame.clone();
        cv::cvtColor(frame_bgr, frame_bgr, cv::COLOR_RGB2BGR);
        
        InferenceManager *im = static_cast<InferenceManager*>(filter->inference_manager);
        im->setDebug(filter->debug);
        auto detections = im->processFrame(frame_bgr);
        
        if (filter->debug) {
            GST_INFO_OBJECT(filter, "Found %zu detections", detections.size());
        }
        
        // Vẽ detections trực tiếp lên frame RGB (không chuyển đổi)
// Vẽ face detections
        for (const auto& det : detections) {
            // Kiểm tra bounding box hợp lệ
            if (det.bbox.width <= 0 || det.bbox.height <= 0) {
                if (filter->debug) {
                    GST_INFO_OBJECT(filter, "Skipping invalid face bbox: w=%d, h=%d", 
                                   det.bbox.width, det.bbox.height);
                }
                continue;
            }
            
            // Đảm bảo bbox nằm trong frame
            cv::Rect safe_bbox = det.bbox;
            safe_bbox.x = std::max(0, safe_bbox.x);
            safe_bbox.y = std::max(0, safe_bbox.y);
            safe_bbox.width = std::min(safe_bbox.width, width - safe_bbox.x);
            safe_bbox.height = std::min(safe_bbox.height, height - safe_bbox.y);
            
            // Kiểm tra lại sau khi clamp
            if (safe_bbox.width <= 0 || safe_bbox.height <= 0) {
                continue;
            }
            
            // Màu xanh lá cho face detection
            cv::Scalar face_color(0, 255, 0);  // Green trong RGB
            
            // Vẽ face bounding box
            cv::rectangle(frame, safe_bbox, face_color, 2);
            
            // Tạo label text
            std::string label_text = "Face: " + std::to_string((int)(det.confidence * 100)) + "%";
            
            // Tính toán vị trí text
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            
            // Vẽ text ở trên face
            cv::Point text_origin(safe_bbox.x, safe_bbox.y - 5);
            if (text_origin.y < text_size.height) {
                text_origin.y = safe_bbox.y + text_size.height + 5;
            }
            
            // Background cho text
            cv::Rect text_bg(text_origin.x, text_origin.y - text_size.height - 2,
                           text_size.width + 4, text_size.height + 4);
            
            // Đảm bảo text nằm trong frame
            if (text_bg.x >= 0 && text_bg.y >= 0 && 
                text_bg.x + text_bg.width <= width && 
                text_bg.y + text_bg.height <= height) {
                
                cv::rectangle(frame, text_bg, face_color, -1);
                cv::putText(frame, label_text, text_origin,
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            }
            
            // Vẽ thêm một chấm ở giữa face để dễ nhận biết
            cv::Point face_center(safe_bbox.x + safe_bbox.width/2, safe_bbox.y + safe_bbox.height/2);
            cv::circle(frame, face_center, 3, cv::Scalar(255, 0, 0), -1);  // Red dot
            
            if (filter->debug) {
                GST_INFO_OBJECT(filter, "Drew face: conf=%.2f at [%d,%d,%d,%d]",
                               det.confidence, safe_bbox.x, safe_bbox.y, 
                               safe_bbox.width, safe_bbox.height);
            }
        }
        
        // Không cần chuyển đổi màu lại vì đã vẽ trực tiếp lên frame RGB
        
    } catch (const std::exception& e) {
        GST_ERROR_OBJECT(filter, "Error: %s", e.what());
        gst_buffer_unmap(buf, &map);
        return GST_FLOW_ERROR;
    }
    
    gst_buffer_unmap(buf, &map);
    return GST_FLOW_OK;
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    facedetect,
    "Face Detection Element",
    plugin_init,
    VERSION,
    "LGPL",
    GST_PACKAGE_NAME,
    GST_PACKAGE_ORIGIN
)
