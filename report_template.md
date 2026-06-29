# Automatic Coin Counting System using YOLO and Computer Vision

## 1. Problem Description

The goal of this project is to implement a Computer Vision system that automatically counts coins in real images. The input is an image containing coins, and the output is a numeric decision: the number of detected coins.

## 2. Team Roles and Task Division

| Team Member | Role | Tasks |
|---|---|---|
| Name 1 | Lead CV Engineer | YOLO integration, detection, decision logic |
| Name 2 | Image Processing Specialist | Image enhancement and segmentation visualization |
| Name 3 | Morphology & Report Lead | Mask cleaning, output visualization, report |
| Name 4 | Data & Testing Engineer | Dataset collection, labeling, experiments |

## 3. Pipeline Design

The project follows the required pipeline:

```text
image -> enhance -> segment -> clean -> detect -> decide
```

## 4. Methods Used

### 4.1 Enhance

The image is converted to LAB color space. CLAHE improves the L channel. Bilateral filtering reduces noise while preserving edges.

### 4.2 Segment

YOLO returns bounding boxes around detected coins. These regions are converted into a binary segmentation mask.

### 4.3 Clean

Detections with low confidence are removed. The mask is cleaned using morphological closing.

### 4.4 Detect

The detection stage uses YOLO object detection. The model is trained on images labeled with one class: `coin`.

### 4.5 Decide

The final decision is calculated as the number of cleaned YOLO detections.

## 5. Results

Insert images from `data/results`.

Example table:

| Image | Expected Count | Detected Count | Result |
|---|---:|---:|---|
| test_01.jpg | 10 | 10 | Correct |
| test_02.jpg | 14 | 13 | One missed coin |
| test_03.jpg | 6 | 6 | Correct |

## 6. Failure Cases

Possible failure cases:

1. Coins are heavily covered by other coins.
2. The image is very blurry.
3. The coin is too small.
4. The confidence threshold is too high.
5. The model was not trained on similar backgrounds or lighting conditions.

## 7. Conclusion

The project implements a complete Computer Vision pipeline for automatic coin counting. The system uses image enhancement, YOLO object detection, mask generation, detection filtering, and automatic decision logic.
