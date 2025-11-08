import argparse
import cv2
from ultralytics import YOLO

def draw_boxes(im, preds):
    for r in preds:
        boxes = r.boxes
        for b in boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            conf = float(b.conf[0])
            cls = int(b.cls[0])
            label = f"{r.names[cls]} {conf:.2f}"
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(im, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return im

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Real-Time Object Detection")
    parser.add_argument("--weights", default="yolov8n.pt", help="Path to trained weights")
    parser.add_argument("--source", default="0", help="Camera index or video path (0 for webcam)")
    args = parser.parse_args()

    print(f"🎥 Loading YOLOv8 model: {args.weights}")
    model = YOLO(args.weights)
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame, imgsz=640, conf=0.25, verbose=False)
        frame = draw_boxes(frame, results)
        cv2.imshow("AutoVision.AI - Real-Time Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
