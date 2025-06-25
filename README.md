Real-Time Object Instance Segmentation with YOLOv8-seg
Overview
This project focuses on real-time object instance segmentation using the Ultralytics YOLOv8-seg model and OpenCV. It's designed to process video streams, identify distinct objects, segment them with unique colors, and visualize the results with bounding boxes and confidence scores. This system provides a robust solution for understanding complex scenes by isolating individual objects, making it valuable for applications requiring precise object localization and shape analysis.

This iteration emphasizes the software and machine learning aspects of real-time vision processing, moving away from embedded systems and robotic control to concentrate on advanced computer vision capabilities.

Abstract
This research presents a system for real-time object instance segmentation leveraging the Ultralytics YOLOv8-seg model for robust and efficient visual scene understanding. The project aims to process video input, perform pixel-level segmentation of individual object instances, and display these results with high confidence and accuracy. This system addresses the need for detailed visual information in various applications, from surveillance to interactive media, by offering precise object boundary identification.

The methodology involves utilizing a pre-trained yolov8n-seg.pt model to analyze video frames. For each detected object, a unique colored segmentation mask is overlaid on the original frame, complemented by bounding boxes, class labels, and confidence scores. The system is configurable to accept different video input sources, demonstrating foundational Computer Vision (CV) and image and video understanding principles.

Preliminary tests demonstrate effective real-time processing and accurate segmentation of multiple object instances within dynamic video environments. The expected outcome is a functioning proof-of-concept for high-fidelity object segmentation that can be adapted for various vision-based analytical tasks.

Features
Real-time Segmentation: Processes video frames in real-time to perform object instance segmentation.

YOLOv8n-seg Model: Utilizes the efficient yolov8n-seg.pt pre-trained model for fast and accurate segmentation.

Mask Visualization: Overlays colored segmentation masks on detected objects, making individual instances clear.

Bounding Box and Labeling: Draws bounding boxes and displays class labels with confidence scores for each detected object.

Configurable Input: Easily change the input video file.

Setup and Usage
To run this project, follow these steps:

Clone the repository:

git clone https://github.com/your-repo/real-time-object-segmentation.git
cd real-time-object-segmentation

Install dependencies:

pip install opencv-python numpy ultralytics

Download the YOLOv8n-seg model:
The script will automatically download yolov8n-seg.pt if it's not present when YOLO('yolov8n-seg.pt') is called.

Place your video file:
Ensure your input video file (e.g., park.mp4) is in the root directory of the project, or update the APP_INPUT_VIDEO_PATH variable in src/real_time_object_segmentation_app.py to point to your video.

Run the application:

python src/real_time_object_segmentation_app.py

Code Structure
.
├── README.md
├── src/
│   └── real_time_object_segmentation_app.py
└── yolov8n-seg.pt (This will be downloaded automatically)
└── park.mp4 (Example input video)

The core logic resides in src/real_time_object_segmentation_app.py, organized into modular classes:

VideoProcessor: Handles video loading, YOLOv8-seg model inference, and applying segmentation overlays to frames.

SegmentationVisualizer: Manages the OpenCV display window and renders the processed frames with performance metrics.

ApplicationCoordinator: Orchestrates the overall application flow, coordinating between the VideoProcessor and SegmentationVisualizer.

NASA 10 Coding Principles
This project adheres to key principles for robust and predictable software design, particularly relevant for real-time systems:

No dynamic memory allocation after initialization: Minimizes runtime overhead.

No recursion: Ensures predictable stack usage.

All loops must have a fixed upper bound: Guarantees real-time predictability for iteration-based processes.

No floating-point arithmetic if not explicitly justified: Prioritizes performance while acknowledging necessary ML operations.

All variables must be explicitly declared and initialized: Promotes clear state and prevents undefined behavior.

Future Enhancements
Performance Optimization: Explore quantization or model pruning for improved inference speed.

Diverse Input Sources: Add support for live camera feeds or network streams.

Custom Model Training: Provide instructions or a pipeline for training YOLOv8-seg on custom datasets.

User Interface: Implement a more interactive GUI for controls and visualizations beyond basic OpenCV windows.
├── .gitignore
├── src/
│   └── real_time_occupancy_mapper_app.py
└── models/
    └── .gitkeep (placeholder)

