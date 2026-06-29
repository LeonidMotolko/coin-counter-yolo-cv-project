from ultralytics import YOLO


def main():
    """
    Train YOLO model.

    Before running:
    1. Put images into dataset/images/train and dataset/images/val.
    2. Put YOLO labels into dataset/labels/train and dataset/labels/val.
    3. Check data.yaml.
    """

    model = YOLO("yolov8n.pt")

    model.train(
        data="Coin Dataset.v1i.yolov8/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        name="coin_counter_yolo",
        patience=15
    )


if __name__ == "__main__":
    main()
