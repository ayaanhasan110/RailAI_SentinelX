import cv2
import os
import time

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit(
        "\nUltralytics is not installed. Run:\n"
        "python -m pip install ultralytics\n"
    )

MODEL_PATH = "yolo11n.pt"
CAMERA_INDEX = 0
CONFIDENCE = 0.35

if not os.path.exists(MODEL_PATH):
    raise SystemExit(
        f"\nModel not found: {MODEL_PATH}\n"
        "Put yolo11n.pt in the same folder as this file.\n"
    )

print("Loading YOLO11n...")
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise SystemExit(
        "\nCould not open camera. Check Windows Camera permissions.\n"
    )

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\nYOLO CAMERA STARTED")
print("Objects will be detected and tracked.")
print("Press Q to quit.")

previous_time = time.time()
fps = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        print("Could not read a camera frame.")
        break

    results = model.track(
        source=frame,
        persist=True,
        conf=CONFIDENCE,
        imgsz=640,
        verbose=False
    )

    annotated = results[0].plot()

    count = len(results[0].boxes) if results[0].boxes is not None else 0

    now = time.time()
    dt = max(now - previous_time, 1e-6)
    instant_fps = 1.0 / dt
    fps = 0.9 * fps + 0.1 * instant_fps if fps else instant_fps
    previous_time = now

    cv2.rectangle(annotated, (10, 10), (320, 82), (0, 0, 0), -1)
    cv2.putText(annotated, f"Objects: {count}", (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(annotated, f"FPS: {fps:.1f}   Q = quit", (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

    cv2.imshow("RailAI Sentinel X - YOLO11n", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Camera stopped.")
