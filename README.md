# Hailo-8 Emotion Detection for Raspberry Pi 5

Real-time emotion detection using Hailo-8 AI accelerator on Raspberry Pi 5. 
Detects 7 emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

## Overview

This project provides a complete pipeline for real-time facial emotion recognition on Raspberry Pi 5 with Hailo-8 HAT+. The model is trained on a custom dataset and optimized for edge deployment with INT8 quantization.

### Performance

- FER2013 benchmark accuracy: 61.7%
- Real-time inference: 30-40 FPS
- Latency: 15-20ms per frame
- Model size: ~8MB

Note that real-world accuracy is typically lower (50-55%) due to domain shift between the training dataset and live camera feeds.

## Requirements

### Hardware
- Raspberry Pi 5 (8GB recommended)
- Hailo-8 HAT+ installed via PCIe
- Camera (USB webcam or Pi Camera Module)

### Software
- Raspberry Pi OS (64-bit)
- Hailo SDK 4.20 or later
- TAPPAS 3.31.0 or later
- Python 3.8+
- GStreamer 1.0+

## Installation

1. Install Hailo SDK and TAPPAS following the official documentation

2. Clone this repository:
```bash
git clone https://github.com/NecheSeTopper/hailo8-realtime-emotion-detection.git
cd hailo8-emotion-detection
```

3. The models are stored with Git LFS. If they don't download automatically:
```bash
git lfs pull
```

4. Make scripts executable:
```bash
chmod +x scripts/*.sh
```

## Usage

Run with camera:
```bash
./scripts/realtime_emotion_camera.sh
```

Run with video file:
```bash
./scripts/realtime_emotion_camera.sh /path/to/video.mp4
```

Test on FER2013 dataset:
```bash
python3 scripts/test_fer2013_hailort.py \
    --model models/emotion.hef \
    --dataset /path/to/fer2013/test
```

## Project Structure

```
hailo8-emotion-detection/
├── models/
│   ├── emotion.hef                    # Emotion classification model
│   └── retinaface_mobilenet_v1.hef   # Face detection model
├── resources/
│   ├── emotion_labels.json            # Label mappings
│   └── emotion_post.so                # Post-processing library
├── scripts/
│   ├── realtime_emotion_camera.sh     # Main inference script
│   └── test_fer2013_hailort.py        # Benchmark testing
└── README.md
```

## Benchmark Results

Per-class performance on FER2013 test set (7,178 samples):

| Emotion  | Samples | Correct | Recall | Precision |
|----------|---------|---------|--------|------------|
| Angry    | 958     | 551     | 57.5%  | 50.2%     |
| Disgust  | 111     | 68      | 61.3%  | 68.0%     |
| Fear     | 1,024   | 354     | 34.6%  | 52.6%     |
| Happy    | 1,774   | 1,348   | 76.0%  | 85.4%     |
| Sad      | 1,247   | 701     | 56.2%  | 44.9%     |
| Surprise | 831     | 666     | 80.1%  | 77.0%     |
| Neutral  | 1,233   | 741     | 60.1%  | 58.8%     |

Overall accuracy: 4,429/7,178 (61.7%)

## Known Limitations

The model performs well on the FER2013 benchmark but has lower accuracy in real-world scenarios due to:

- Domain shift: Differences in lighting, camera quality, and facial poses
- Expression dynamics: The model struggles with transitional states and talking

Common issues:
- Over-predicts Surprise when mouth is open (talking, yawning)
- Lower performance on Disgust and Fear (underrepresented in training data)

Best results are achieved with:
- Frontal face position (within 15 degrees)
- Good, even lighting
- Camera resolution of 720p or higher
- Stationary subject

## Troubleshooting

If models are not found:
```bash
git lfs install
git lfs pull
```

If Hailo device is not detected:
```bash
lspci | grep Hailo
hailortcli fw-control identify
```

For GStreamer issues:
```bash
source /opt/hailo/tappas/scripts/setup_env.sh
gst-inspect-1.0 hailonet
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- FER2013 dataset by Goodfellow et al., 2013
- Hailo AI SDK and community
- Raspberry Pi Foundation
- TAPPAS framework
