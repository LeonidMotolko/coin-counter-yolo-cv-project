import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def collect_images(input_path: str):
    input_path = Path(input_path)

    if input_path.is_file():
        return [input_path]

    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in input_path.iterdir() if p.suffix.lower() in valid_ext])


# ============================================================
# 1. ENHANCE
# ============================================================

def enhance_image(image):
    """
    Stage 1: Enhance.
    Improve contrast before detection.

    Methods:
    - LAB color space;
    - CLAHE on L channel;
    - light denoising with bilateral filter.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    enhanced = cv2.bilateralFilter(enhanced, 5, 40, 40)

    return enhanced


# ============================================================
# 2. SEGMENT
# ============================================================

def create_segmentation_mask(image_shape, boxes):
    """
    Stage 2: Segment.
    YOLO gives object regions as bounding boxes.
    For project visualization, each detected coin region is written to a binary mask.
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for box in boxes:
        x1, y1, x2, y2 = box["xyxy"]
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    return mask


# ============================================================
# 3. CLEAN
# ============================================================

def clean_detections(boxes, confidence_threshold=0.35):
    """
    Stage 3: Clean.
    Remove weak detections using confidence threshold.
    """
    return [box for box in boxes if box["confidence"] >= confidence_threshold]


def clean_mask(mask):
    """
    Stage 3: Clean visualization mask with morphology.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cleaned


# ============================================================
# 4. DETECT
# ============================================================

def detect_coins(model, enhanced_image, confidence_threshold=0.35, image_size=640):
    """
    Stage 4: Detect.
    Detect coins using trained YOLO model.
    """
    results = model.predict(
        source=enhanced_image,
        imgsz=image_size,
        conf=confidence_threshold,
        verbose=False
    )

    boxes = []

    if not results:
        return boxes

    result = results[0]

    if result.boxes is None:
        return boxes

    for b in result.boxes:
        xyxy = b.xyxy[0].cpu().numpy().astype(int)
        confidence = float(b.conf[0].cpu().numpy())
        class_id = int(b.cls[0].cpu().numpy())

        x1, y1, x2, y2 = xyxy.tolist()

        boxes.append({
            "xyxy": [x1, y1, x2, y2],
            "confidence": confidence,
            "class_id": class_id,
            "class_name": "coin"
        })

    return boxes


# ============================================================
# 5. DECIDE
# ============================================================

def decide(cleaned_boxes):
    """
    Stage 5: Decide.
    The final automatic decision is the number of detected coins.
    """
    count = len(cleaned_boxes)

    if count == 0:
        label = "No coins detected"
    elif count == 1:
        label = "Detected 1 coin"
    else:
        label = f"Detected {count} coins"

    return count, label


# ============================================================
# OUTPUT
# ============================================================

def draw_result(image, boxes, decision_text):
    result = image.copy()

    for i, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = box["xyxy"]
        conf = box["confidence"]

        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            result,
            f"{i}: {conf:.2f}",
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    banner_w = min(image.shape[1] - 20, 620)
    cv2.rectangle(result, (10, 10), (10 + banner_w, 62), (255, 255, 255), -1)

    cv2.putText(
        result,
        decision_text,
        (22, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    return result


def save_decision(output_dir, image_name, count, decision_text, boxes):
    path = Path(output_dir) / f"{image_name}_decision.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Final decision: {decision_text}\n")
        f.write(f"Coin count: {count}\n\n")
        f.write("Detected objects:\n")

        for i, box in enumerate(boxes, start=1):
            x1, y1, x2, y2 = box["xyxy"]
            f.write(
                f"{i}. class=coin, confidence={box['confidence']:.3f}, "
                f"box=({x1}, {y1}, {x2}, {y2})\n"
            )


def process_image(model, image_path, output_dir, confidence_threshold=0.35, image_size=640):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    image_name = Path(image_path).stem
    image_output_dir = Path(output_dir) / image_name
    ensure_dir(str(image_output_dir))

    # Full required pipeline.
    enhanced = enhance_image(image)
    raw_boxes = detect_coins(model, enhanced, confidence_threshold, image_size)
    cleaned_boxes = clean_detections(raw_boxes, confidence_threshold)
    segmentation_mask = create_segmentation_mask(image.shape, raw_boxes)
    cleaned_mask = create_segmentation_mask(image.shape, cleaned_boxes)
    cleaned_mask = clean_mask(cleaned_mask)
    count, decision_text = decide(cleaned_boxes)
    detection_result = draw_result(image, cleaned_boxes, decision_text)

    cv2.imwrite(str(image_output_dir / f"{image_name}_01_original.jpg"), image)
    cv2.imwrite(str(image_output_dir / f"{image_name}_02_enhanced.jpg"), enhanced)
    cv2.imwrite(str(image_output_dir / f"{image_name}_03_segmentation_mask.jpg"), segmentation_mask)
    cv2.imwrite(str(image_output_dir / f"{image_name}_04_cleaned_mask.jpg"), cleaned_mask)
    cv2.imwrite(str(image_output_dir / f"{image_name}_05_detection_result.jpg"), detection_result)

    save_decision(image_output_dir, image_name, count, decision_text, cleaned_boxes)

    return {
        "image": str(image_path),
        "decision": decision_text,
        "count": count,
        "output_dir": str(image_output_dir)
    }


def main():
    parser = argparse.ArgumentParser(description="YOLO Coin Counter CV Project")
    parser.add_argument("--input", required=True, help="Path to image or folder with test images")
    parser.add_argument("--output", default="data/results", help="Output folder")
    parser.add_argument("--model", default="models/best.pt", help="Path to trained YOLO model")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO image size")

    args = parser.parse_args()

    if not Path(args.model).exists():
        raise FileNotFoundError(
            f"Model not found: {args.model}\n"
            "Train the model first or put best.pt into the models folder."
        )

    ensure_dir(args.output)

    model = YOLO(args.model)
    images = collect_images(args.input)

    if not images:
        print("No images found.")
        return

    print("YOLO Coin Counter")
    print("=" * 40)

    summary_path = Path(args.output) / "summary.txt"

    with open(summary_path, "w", encoding="utf-8") as summary:
        for image_path in images:
            result = process_image(
                model=model,
                image_path=image_path,
                output_dir=args.output,
                confidence_threshold=args.conf,
                image_size=args.imgsz
            )

            print(f"{Path(result['image']).name}: {result['decision']}")
            print(f"Saved to: {result['output_dir']}")
            print("-" * 40)

            summary.write(f"{Path(result['image']).name}: {result['decision']}\n")

    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
