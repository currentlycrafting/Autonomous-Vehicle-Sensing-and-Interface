# Real-Time Object Instance Segmentation with YOLOv8-seg

## Overview

This project focuses on real-time object instance segmentation using the Ultralytics YOLOv8-seg model and OpenCV. It's designed to process video streams, identify distinct objects, segment them with unique colors, and visualize the results with bounding boxes and confidence scores.

This system provides a robust solution for understanding complex scenes by isolating individual objects, making it valuable for applications requiring precise object localization and shape analysis.

This iteration emphasizes the software and machine learning aspects of real-time vision processing, moving away from embedded systems and robotic control to concentrate on advanced computer vision capabilities.

---

## Abstract

This research presents a system for real-time object instance segmentation leveraging the Ultralytics YOLOv8-seg model for robust and efficient visual scene understanding. The project aims to process video input, perform pixel-level segmentation of individual object instances, and display these results with high confidence and accuracy.

The system uses a pre-trained `yolov8n-seg.pt` model to analyze video frames. For each detected object, a unique colored segmentation mask is overlaid on the original frame, with bounding boxes, class labels, and confidence scores. It supports configurable input video sources and demonstrates foundational computer vision (CV) principles.

### Key Outcomes

- Effective real-time segmentation
- Accurate detection of multiple object instances
- Adaptability to various video inputs

---

## Features

- ✅ **Real-Time Segmentation**: Processes video frames live using object instance segmentation.
- 🧠 **YOLOv8n-seg Model**: Utilizes a lightweight, pre-trained model for high-speed inference.
- 🎨 **Mask Visualization**: Distinct segmentation masks overlay on each object.
- 🏷️ **Bounding Boxes and Labels**: Includes confidence scores and class names.
- 🎥 **Configurable Input**: Easily switch between different video files.

---

## Setup & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/real-time-object-segmentation.git
cd real-time-object-segmentation

├── src/
│   └── real_time_occupancy_mapper_app.py
└── models/
    └── .gitkeep (placeholder)

