#!/bin/bash
# Real-time emotion detection with camera or video file

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"  # Parent directory
FACE_HEF="$PROJECT_DIR/models/retinaface_mobilenet_v1.hef"
EMOTION_HEF="$PROJECT_DIR/models/emotion.hef"
# Allow user to specify TAPPAS installation path
POSTPROC_DIR="${TAPPAS_POSTPROC_DIR:-/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes}"
EMOTION_POST_LIB="$PROJECT_DIR/resources/emotion_post.so"
FACE_POST_LIB="$POSTPROC_DIR/libface_detection_post.so"
CROP_LIB="$POSTPROC_DIR/cropping_algorithms/libdetection_croppers.so"

# Check if video file argument is provided
VIDEO_FILE=""
if [ $# -gt 0 ]; then
    VIDEO_FILE="$1"
    if [ ! -f "$VIDEO_FILE" ]; then
        echo " ERROR: Video file not found: $VIDEO_FILE"
        exit 1
    fi
fi

# Validate files
if [ -n "$VIDEO_FILE" ]; then
    echo "==================================================================="
    echo "   Emotion Detection from Video File"
    echo "==================================================================="
    echo " Source: $VIDEO_FILE"
else
    echo "==================================================================="
    echo "   Real-Time Emotion Detection with Camera"
    echo "==================================================================="
    echo " Camera: Camera"
fi
echo ""

if [ ! -f "$FACE_HEF" ]; then
    echo " ERROR: Face detection HEF not found: $FACE_HEF"
    exit 1
fi

if [ ! -f "$EMOTION_HEF" ]; then
    echo " ERROR: Emotion HEF not found: $EMOTION_HEF"
    exit 1
fi

if [ ! -f "$EMOTION_POST_LIB" ]; then
    echo " ERROR: Emotion post-process library not found: $EMOTION_POST_LIB"
    exit 1
fi

echo " All files validated"
echo ""
echo "Configuration:"
echo "  TAPPAS Post-processing: $POSTPROC_DIR"
echo "  Face Detection HEF: $FACE_HEF"
echo "  Emotion HEF: $EMOTION_HEF"
echo "  Emotion Post-process: $EMOTION_POST_LIB"
echo ""
if [ -n "$VIDEO_FILE" ]; then
    echo "Press Ctrl+C to stop video playback"
else
    echo "Press Ctrl+C to stop"
fi
echo "==================================================================="
echo ""

# Enable debug output
export EMOTION_DEBUG=1

# Choose source based on input
if [ -n "$VIDEO_FILE" ]; then
    # Video file pipeline
    SOURCE="filesrc location=\"$VIDEO_FILE\" ! decodebin ! videoconvert ! video/x-raw,format=RGB ! videoscale ! video/x-raw,width=1280,height=736"
else
    # Real-time pipeline with camera source
    SOURCE="libcamerasrc ! video/x-raw,format=YUY2,width=1280,height=720,framerate=30/1 ! videoconvert ! videoscale ! video/x-raw,width=1280,height=736 ! videoconvert ! video/x-raw,format=RGB"
fi

# Choose video sink based on display availability
if [ -n "$DISPLAY" ] || [ -n "$WAYLAND_DISPLAY" ]; then
    VIDEO_SINK="autovideosink"
else
    VIDEO_SINK="fakesink"
    echo "  No display detected, using fakesink for headless operation"
fi

# Run the pipeline
gst-launch-1.0 -v \
    $SOURCE ! \
    tee name=t hailomuxer name=hmux \
    \
    t. ! queue leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! hmux. \
    t. ! queue leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! \
        hailonet hef-path="$FACE_HEF" scheduling-algorithm=1 vdevice-group-id=1 force-writable=true ! \
        queue leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! \
        hailofilter so-path="$FACE_POST_LIB" function-name=retinaface qos=false ! \
        queue leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! \
        hmux. \
    \
    hmux. ! queue leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! \
    hailocropper \
        so-path="$CROP_LIB" \
        function-name=all_detections \
        internal-offset=true \
        use-letterbox=false \
        name=cropper \
    \
    hailoaggregator name=agg flatten-detections=false \
    \
    cropper.src_0 ! \
        queue leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! \
        agg.sink_0 \
    \
    cropper.src_1 ! \
        queue max-size-buffers=3 leaky=downstream ! \
        video/x-raw,pixel-aspect-ratio=1/1,format=RGB ! \
        videoconvert n-threads=2 qos=false ! \
        videoscale method=2 ! \
        video/x-raw,width=224,height=224 ! \
        videoconvert ! \
        video/x-raw,format=RGB ! \
        hailonet hef-path="$EMOTION_HEF" scheduling-algorithm=1 vdevice-group-id=1 force-writable=true ! \
        hailofilter so-path="$EMOTION_POST_LIB" function-name=emotion_mobileNetV2 qos=false ! \
        queue max-size-buffers=3 leaky=downstream ! \
        agg.sink_1 \
    \
    agg. ! \
    queue leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! \
    hailooverlay qos=false show-confidence=true ! \
    queue leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! \
    videoconvert ! \
    fpsdisplaysink video-sink="$VIDEO_SINK" sync=false text-overlay=true 2>&1

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo " Pipeline completed successfully"
else
    echo " Pipeline failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
