import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 Model for Object Detection")
    parser.add_argument("--data", required=True, help="Path to dataset YAML file")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model type (e.g., yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--img", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    args = parser.parse_args()

    print(f"🔧 Starting training with model: {args.model}")
    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.img, batch=args.batch)

if __name__ == "__main__":
    main()
