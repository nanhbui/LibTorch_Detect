#pragma once

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/base/gstbasetransform.h>

G_BEGIN_DECLS

#define GST_TYPE_FACE_DETECT (gst_face_detect_get_type())
G_DECLARE_FINAL_TYPE(GstFaceDetect, gst_face_detect, GST, FACE_DETECT, GstBaseTransform)

struct _GstFaceDetect {
    GstBaseTransform base;
    
    gchar *model_path;
    gboolean passthrough;
    gboolean debug;
    
    GstVideoInfo video_info;
    void *inference_manager;
};

G_END_DECLS
