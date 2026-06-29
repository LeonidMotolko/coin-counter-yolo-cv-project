# YOLO Coin Counter CV Project

## Project idea

This project implements an automatic coin counting system using Computer Vision and YOLO object detection.

Input:

```text
real image with coins
```

Output:

```text
number of detected coins
```

Example:

```text
Detected 14 coins
```

## Required pipeline

The project still follows the required course pipeline:

```text
image -> enhance -> segment -> clean -> detect -> decide
```

### 1. Enhance

The input image is enhanced using:

- LAB color space;
- CLAHE;
- bilateral filtering.

### 2. Segment

YOLO detected object regions are converted into a binary mask.

### 3. Clean

Weak detections are removed using confidence threshold.  
The binary mask is cleaned with morphology.

### 4. Detect

YOLO detects all visible coins and returns bounding boxes.

### 5. Decide

The final decision is the number of detected coins.

## Install

In PyCharm terminal:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset preparation

You need to collect and label images.

Recommended:

- 50-100 photos minimum;
- 100-300 photos is better;
- different numbers of coins;
- different backgrounds;
- different lighting;
- photos from your phone.

## How to label

Use Roboflow or LabelImg.

### Roboflow easiest way

1. Go to Roboflow.
2. Create new Object Detection project.
3. Class name:

```text
coin
```

4. Upload images.
5. Draw boxes around every coin.
6. Split dataset into train/valid.
7. Export as YOLOv8 format.
8. Replace this project's `dataset` folder with exported files.

Expected structure:

```text
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
```

Each image should have a matching `.txt` label file.

## Train model

Run:

```powershell
python train.py
```

After training, copy:

```text
runs/detect/coin_counter_yolo/weights/best.pt
```

to:

```text
models/best.pt
```

## Run detection

Put test images into:

```text
data/test_images
```

Run:

```powershell
python main.py --input data/test_images --output data/results --model models/best.pt
```

If it detects too many coins:

```powershell
python main.py --input data/test_images --output data/results --model models/best.pt --conf 0.55
```

If it misses coins:

```powershell
python main.py --input data/test_images --output data/results --model models/best.pt --conf 0.20
```

## Output files

For each image the system saves:

```text
01_original.jpg
02_enhanced.jpg
03_segmentation_mask.jpg
04_cleaned_mask.jpg
05_detection_result.jpg
decision.txt
```

## Important

This ZIP does not include a trained model because the model must be trained on your own labeled dataset.

The code is complete. To make it work, you need:

```text
models/best.pt
```

after training.
