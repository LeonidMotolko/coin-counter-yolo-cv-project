# Project Proposal: Automatic Coin Counting System using YOLO and Computer Vision

## Problem Description

The goal of this project is to build an automatic Computer Vision system that counts coins in real images. Coin counting is a real visual task because the input is an image and the output is a structured numeric result.

The system detects every visible coin and returns the total number of coins.

## Input and Output

Input:

- real image containing coins.

Output:

- enhanced image;
- segmentation mask;
- cleaned mask;
- detection result with bounding boxes;
- final decision with coin count.

Example final decision:

```text
Detected 14 coins
```

## Pipeline

The system follows the required five-stage pipeline.

### 1. Enhance

The input image is improved using LAB color space, CLAHE, and bilateral filtering.

### 2. Segment

YOLO object regions are converted into a binary segmentation mask.

### 3. Clean

Weak detections are removed using a confidence threshold. The mask is cleaned using morphological operations.

### 4. Detect

Coins are detected using a YOLO object detection model trained on labeled coin images.

### 5. Decide

The final decision is the number of detected coins.

## Dataset

The team will collect its own dataset of real coin images. Each coin will be labeled with a bounding box and the class `coin`.

## Expected Result

The system should correctly count coins on real test images and save all required stage-by-stage outputs.
