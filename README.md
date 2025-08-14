# Peatland Navigation Project

## Overview

This project develops an advanced computer vision system for autonomous navigation in peatland environments. Using multiple deep learning models, it creates a comprehensive navigation assistance system that can identify paths, obstacles, and terrain features while providing real-world measurements for safe traversal.

### Key Objectives

- Safe and efficient path identification in challenging peatland terrain
- Real-time navigation assistance with distance measurements
- Environmental preservation through accurate path following
- Robust operation in varying weather and lighting conditions

### Core Capabilities

- Path detection and segmentation
- Navigation marker recognition
- Real-world distance measurements
- Terrain classification
- Obstacle detection and avoidance

## Technical Architecture

### 1. Multi-Model Vision Pipeline

The system integrates four specialized deep learning models:

1. **Binary Path Segmentation (U-Net)**

   - Primary path identification
   - ResNet34 backbone for robust feature extraction
   - Real-time segmentation capability
   - Initial path proposal generation

2. **Segment Anything Model (SAM)**

   - High-precision boundary refinement
   - Vision Transformer architecture
   - Point-prompt based refinement
   - Accurate path delineation

3. **Object Detection (YOLOv11)**

   - Navigation marker identification
   - Multiple object classes (benches, cones, signs)
   - Real-time detection performance
   - Distance-based measurements

4. **Depth Estimation (MiDaS)**
   - Real-world distance calculation
   - Path width measurement
   - Obstacle distance estimation
   - Spatial awareness enhancement

### 2. Environmental Understanding

The system classifies terrain into five categories:

- **Paths**: Primary navigation routes
- **Natural Ground**: Traversable terrain
- **Trees**: Major obstacles and landmarks
- **Vegetation**: Secondary terrain features
- **Background**: Non-relevant areas

## Project Structure

```
├── data/
│   ├── detection/          # YOLO format detection datasets
│   ├── processed/          # Processed datasets for training
│   ├── segmentation/       # Semantic segmentation datasets
│   └── video/             # Video data and frames
├── data-engineering/       # Data preparation scripts
└── training/              # Training and inference notebooks
```

## Features

- Path Detection using YOLOv11
- Binary Path Segmentation using U-Net
- Multi-class Semantic Segmentation using:
  - U-Net with ResNet34 backbone
  - DinoV2 with custom decoder
- Video Frame Processing and Visualization

## Setup and Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Dataset Structure

- Detection data: YOLO format with classes (Bench, Cones, Signs)
- Segmentation data: Binary (Path/Background) and Multi-class (Path, Natural Ground, Tree, Vegetation)

## Training

### Detection

Use `detection_training.ipynb` for training object detection models.

### Segmentation

Three approaches available:

1. Binary Segmentation (`segmentation_binary_training.ipynb`)
2. Multi-class U-Net (`segmentation_training.ipynb`)
3. DinoV2-based Segmentation (`segementation_dino_training.ipynb`)

## Inference

- `inference_detection.ipynb`: Object detection inference
- `inference_binary_segmentation.ipynb`: Binary path segmentation
- `segmentation_inference.ipynb`: Multi-class segmentation
- `inference_with_depth.ipynb`: Advanced inference with depth estimation

## Evaluation

Evaluation notebooks are provided for each approach:

- `evaluate_detection.ipynb`
- `evaluate_binary_segmentation.ipynb`
- `evaluate_segmentation.ipynb`
- `evaluate_segmentation_dino.ipynb`

## Dataset Organization

### Detection Dataset

- Format: YOLO-style annotations
- Classes: Bench, Cones, Signs
- Split: train/val/test
- Includes negative samples for robustness

### Segmentation Dataset

1. **Binary Dataset**

   - Two classes: Path/Non-path
   - Focused on path identification
   - High-precision annotations

2. **Multi-class Dataset**
   - Five terrain classes
   - Pixel-wise annotations
   - Balanced class distribution

## Training Pipeline

### 1. Object Detection

- YOLOv11 architecture
- Transfer learning from COCO
- Custom augmentation pipeline
- Negative sample integration

### 2. Binary Segmentation

- U-Net with ResNet34
- Binary cross-entropy loss
- Real-time inference focus
- Temporal consistency optimization

### 3. Multi-class Segmentation

- Choice of architectures:
  - U-Net with ResNet34
  - DinoV2 with custom decoder
- Class-balanced loss
- Advanced augmentation strategy

## Inference and Deployment

### Video Processing

- Frame rate: 10 FPS (optimized)
- Resolution: 640x480
- Real-time capable processing
- Temporal smoothing integration

### Output Features

- Path boundary visualization
- Distance measurements
- Object detection with confidence
- Path width estimation
- Navigation markers

## Evaluation Metrics

### Detection Performance

- mAP50-95
- Precision-Recall curves
- Class-wise accuracy
- Real-time processing speed

### Segmentation Quality

- IoU (Intersection over Union)
- Pixel-wise accuracy
- Class-wise metrics
- Temporal consistency



