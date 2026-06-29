import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def collect_images(input_path: str):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([
        path for path in input_path.iterdir()
        if path.suffix.lower() in valid_extensions
    ])


# ============================================================
# 1. Enhance
# ============================================================

def enhance_image(image):
    """
    Stage 1: Enhance.
    Improve image quality before segmentation and detection.
    Methods: LAB color space, CLAHE, bilateral filtering.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=40, sigmaSpace=40)

    return enhanced


# ============================================================
# 2. Segment
# ============================================================

def segment_image(enhanced_image):
    """
    Stage 2: Segment.
    Real OpenCV segmentation before YOLO detection.
    Creates a binary mask of visually important regions.
    """
    gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)

    segmentation_mask = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        5
    )

    return segmentation_mask


# ============================================================
# 3. Clean
# ============================================================

def clean_mask(segmentation_mask):
    """
    Stage 3: Clean.
    Clean the segmentation mask using morphology.
    Opening removes small noise; closing fills small gaps.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    cleaned_mask = cv2.morphologyEx(
        segmentation_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    cleaned_mask = cv2.morphologyEx(
        cleaned_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    return cleaned_mask


def clean_detections(boxes, confidence_threshold=0.35):
    """
    Additional post-detection filtering.
    This is not the main Clean pipeline stage.
    The main Clean stage is clean_mask(), which happens before detection.
    """
    return [box for box in boxes if box["confidence"] >= confidence_threshold]


# ============================================================
# 4. Detect
# ============================================================

def detect_coins(model, enhanced_image, confidence_threshold=0.35, image_size=640):
    """
    Stage 4: Detect.
    Detect coins using the trained YOLO model.
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

    for detected_box in result.boxes:
        xyxy = detected_box.xyxy[0].cpu().numpy().astype(int)
        confidence = float(detected_box.conf[0].cpu().numpy())
        class_id = int(detected_box.cls[0].cpu().numpy())

        x1, y1, x2, y2 = xyxy.tolist()

        boxes.append({
            "xyxy": [x1, y1, x2, y2],
            "confidence": confidence,
            "class_id": class_id,
            "class_name": "coin"
        })

    return boxes


# ============================================================
# 5. Decide
# ============================================================

def decide(cleaned_boxes):
    """
    Stage 5: Decide.
    The final automatic decision is the number of detected coins.
    """
    count = len(cleaned_boxes)

    if count == 0:
        decision_text = "No coins detected"
    elif count == 1:
        decision_text = "Detected 1 coin"
    else:
        decision_text = f"Detected {count} coins"

    return count, decision_text


# ============================================================
# Visualization and output
# ============================================================

def draw_detection_result(image, boxes, decision_text):
    result = image.copy()

    for index, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = box["xyxy"]
        confidence = box["confidence"]

        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            result,
            f"{index}: {confidence:.2f}",
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    banner_width = min(image.shape[1] - 20, 620)
    cv2.rectangle(result, (10, 10), (10 + banner_width, 62), (255, 255, 255), -1)

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
    decision_path = Path(output_dir) / f"{image_name}_decision.txt"

    with open(decision_path, "w", encoding="utf-8") as file:
        file.write(f"Final decision: {decision_text}\n")
        file.write(f"Coin count: {count}\n\n")
        file.write("Detected objects:\n")

        for index, box in enumerate(boxes, start=1):
            x1, y1, x2, y2 = box["xyxy"]
            file.write(
                f"{index}. class=coin, "
                f"confidence={box['confidence']:.3f}, "
                f"box=({x1}, {y1}, {x2}, {y2})\n"
            )


def process_image(model, image_path, output_dir, confidence_threshold=0.35, image_size=640):
    """
    Process one image using the required pipeline order:
    image -> enhance -> segment -> clean -> detect -> decide
    """
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    image_name = Path(image_path).stem
    image_output_dir = Path(output_dir) / image_name
    ensure_dir(str(image_output_dir))

    original_image = image
    enhanced_image = enhance_image(original_image)
    segmentation_mask = segment_image(enhanced_image)
    cleaned_mask = clean_mask(segmentation_mask)

    raw_boxes = detect_coins(
        model=model,
        enhanced_image=enhanced_image,
        confidence_threshold=confidence_threshold,
        image_size=image_size
    )

    cleaned_boxes = clean_detections(
        boxes=raw_boxes,
        confidence_threshold=confidence_threshold
    )

    count, decision_text = decide(cleaned_boxes)
    detection_result = draw_detection_result(original_image, cleaned_boxes, decision_text)

    cv2.imwrite(str(image_output_dir / f"{image_name}_01_original.jpg"), original_image)
    cv2.imwrite(str(image_output_dir / f"{image_name}_02_enhanced.jpg"), enhanced_image)
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


# ============================================================
# Program entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="YOLO Coin Counter CV Project")

    parser.add_argument("--input", required=True, help="Path to one image or folder with test images")
    parser.add_argument("--output", default="data/results", help="Folder where results will be saved")
    parser.add_argument("--model", default="models/best.pt", help="Path to trained YOLO model")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold for YOLO detections")
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

    with open(summary_path, "w", encoding="utf-8") as summary_file:
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

            summary_file.write(f"{Path(result['image']).name}: {result['decision']}\n")

    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
