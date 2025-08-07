# Peatland Navigation Project

## Overview

This project focuses on computer vision-based navigation in peatland environments. It implements multiple deep learning approaches for path detection and semantic segmentation, helping to identify navigable paths, vegetation, trees, and natural ground in peatland areas.

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

## Model Outputs

Final video outputs can be found in `training/final_video_output/`
