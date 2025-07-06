#include <gst/gst.h>
#include <iostream>

int main(int argc, char *argv[]) {
    gst_init(&argc, &argv);
    
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <video_source>" << std::endl;
        std::cerr << "Example video sources:" << std::endl;
        std::cerr << "  Webcam: v4l2src device=/dev/video0" << std::endl;
        std::cerr << "  Video file: filesrc location=test.mp4 ! decodebin" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    std::string video_source = argv[2];
    
    std::string pipeline_str = 
        video_source + " ! videoconvert ! facedetect model-path=" + model_path + " ! videoconvert ! autovideosink";
    
    GstElement *pipeline = gst_parse_launch(pipeline_str.c_str(), NULL);
    if (!pipeline) {
        std::cerr << "Failed to create pipeline" << std::endl;
        return 1;
    }
    
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    
    GMainLoop *loop = g_main_loop_new(NULL, FALSE);
    g_main_loop_run(loop);
    
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    g_main_loop_unref(loop);
    
    return 0;
}
