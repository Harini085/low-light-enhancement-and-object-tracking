# Low-Light Enhancement and Object Tracking

## Overview

This project enhances low-light videos and performs real-time object detection and tracking using a deep learning-based low-light enhancement model and YOLO object detection.

The enhancement model improves the visibility of dark scenes before object detection, resulting in better tracking performance under challenging lighting conditions.

---

## Features

- Low-light video enhancement
- Real-time object detection
- Object tracking
- YOLO-based detection
- Supports video input

---

## Project Structure

```
low-light-enhancement-and-object-tracking/
│
├── video_tracking_v5.py
├── README.md
└── samples/
    └── input/
        └── vdo1_i.mp4
```

---

## Requirements

- Python 3.10+
- PyTorch
- OpenCV
- Ultralytics (YOLO)
- NumPy

Install the required packages:

```bash
pip install torch torchvision ultralytics opencv-python numpy
```

---

## Model Files

The trained model weights are **not included** in this repository because they exceed GitHub's file size limit.

Place the following files in the project directory before running the project:

- `iccv_fuzzy_net_epoch_120.pth`
- `yolo12n.pt` *(or `yolo12s.pt` if using the small model)*

---

## Running the Project

Execute the following command:

```bash
python video_tracking_v5.py
```

If necessary, update the input video path inside `video_tracking_v5.py`.

---

## Sample Input

Sample video:

```
samples/input/vdo1_i.mp4
```

---

## Applications

- Smart Surveillance
- Night-Time Monitoring
- Traffic Analysis
- Security Systems
- Intelligent Video Analytics

---

## Future Improvements

- Live webcam support
- Multi-object tracking improvements
- GPU optimization
- Custom YOLO model training
- User-friendly interface

---

## Author

**Harini S**
