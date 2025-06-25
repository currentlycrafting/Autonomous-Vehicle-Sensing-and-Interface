# Real-Time Vision-Based Occupancy Mapping for Autonomous Navigation Using Low-Cost Multi-Camera Systems

---

## Overview

This project explores how **low-cost, vision-only autonomous systems** can achieve **human-trustworthy spatial awareness** using real-time **multi-camera input** and machine learning. We aim to build a **self-navigating model vehicle** that uses camera data to generate **Bird's Eye View (BEV) occupancy maps**, enabling it to identify free space and obstacles for autonomous navigation.

This system offers an alternative to expensive LiDAR/radar setups by relying solely on camera input, making it suitable for scalable, cost-effective autonomous platforms.

---

## System Architecture
[ Multi-Camera Input (Front, Left, Right) ]
            ↓
  [ Stereo Image Processing + Depth Estimation ]
            ↓
      [ CNN-Based Occupancy Segmentation ]
            ↓
      [ Bird's Eye View Heatmap Generation ]
            ↓
[ Rule-Based Controller ] ---> [ Motor Control / Stop Logic ]
            ↓
      [ Dashboard Visualizer ]

--

# Abstract
This research investigates how low-cost, vision-only autonomous systems can achieve human-trustworthy spatial awareness using real-time multi-camera input and machine learning. The project aims to build a self-navigating, camera-powered model vehicle capable of detecting free space and obstacles using a simplified Bird's Eye View (BEV) occupancy map derived from stereo vision and a trained spatial segmentation model. This system addresses the growing demand for safer and more transparent camera-based navigation methods in customer-facing autonomous technologies.

The methodology includes mounting three cameras (front, left, and right) on a mobile vehicle chassis, capturing labeled driving scene images across a controlled test course, and training a lightweight Convolutional Neural Network (CNN) to generate occupancy maps from those views. Depth will be approximated using stereo image pairs, and resulting BEV maps will be visualized as live-updating heatmaps on a local dashboard. A rule-based controller will interpret the BEV maps to initiate movement or stopping decisions in real time.

Preliminary tests on simulated data demonstrate effective detection of occupied versus navigable zones. The expected outcome is a functioning proof-of-concept that links spatial vision modeling with real-world action. Future work will explore improved fusion methods and confidence-aware learning.


.
├── README.md
├── .gitignore
├── src/
│   └── real_time_occupancy_mapper_app.py
└── models/
    └── .gitkeep (placeholder)

