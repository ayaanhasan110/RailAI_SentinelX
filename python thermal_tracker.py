import cv2
import time
import math
from collections import defaultdict, deque

from ultralytics import YOLO


# ============================================================
# AI MULTI-OBJECT DETECTION + TRACKING
# CPU FRIENDLY VERSION
# ============================================================

print("=" * 60)
print(" AI MULTI-OBJECT VISION SYSTEM")
print("=" * 60)

print("Loading AI model...")

# Nano model = much lighter for an i3 CPU
model = YOLO("yolo11n.pt")

print("AI model loaded.")
print("Starting camera...")


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()


# Keep resolution modest for CPU performance
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# ============================================================
# TRACKING SETTINGS
# ============================================================

# Store recent positions for each object
position_history = defaultdict(lambda: deque(maxlen=10))

# Store previous timestamp for speed calculations
time_history = {}

# FPS calculation
previous_time = time.time()

# Minimum AI confidence
CONFIDENCE = 0.40

# Detection image size
IMAGE_SIZE = 416


# ============================================================
# SPEED CALIBRATION
# ============================================================

# IMPORTANT:
#
# Leave this as None if you don't have a known physical
# distance in your camera scene.
#
# In that case the program reports PIXELS/SECOND.
#
# Example:
#
# REAL_METERS_PER_PIXEL = 0.005
#
# would mean:
# 200 pixels = 1 meter
#
# This requires proper camera calibration.

REAL_METERS_PER_PIXEL = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_pixel_speed(track_id):

    history = position_history[track_id]

    if len(history) < 2:
        return 0.0

    (x1, y1, t1) = history[-2]
    (x2, y2, t2) = history[-1]

    distance_pixels = math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )

    delta_time = t2 - t1

    if delta_time <= 0:
        return 0.0

    return distance_pixels / delta_time


def pixel_speed_to_real_speed(pixel_speed):

    if REAL_METERS_PER_PIXEL is None:
        return None

    meters_per_second = (
        pixel_speed * REAL_METERS_PER_PIXEL
    )

    km_per_hour = meters_per_second * 3.6

    return km_per_hour


def get_speed_text(track_id):

    pixel_speed = calculate_pixel_speed(track_id)

    real_speed = pixel_speed_to_real_speed(
        pixel_speed
    )

    if real_speed is None:

        return f"{pixel_speed:.1f} px/s"

    return f"{real_speed:.1f} km/h"


# ============================================================
# MAIN LOOP
# ============================================================

print()
print("SYSTEM ONLINE")
print("Press Q to quit.")
print("Press S to save a screenshot.")
print()


while True:

    success, frame = camera.read()

    if not success:

        print("ERROR: Could not read camera frame.")

        break


    current_time = time.time()


    # ========================================================
    # AI OBJECT DETECTION + TRACKING
    # ========================================================

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        verbose=False
    )


    # ========================================================
    # FPS
    # ========================================================

    elapsed = current_time - previous_time

    if elapsed > 0:

        fps = 1.0 / elapsed

    else:

        fps = 0

    previous_time = current_time


    # ========================================================
    # PROCESS AI RESULTS
    # ========================================================

    object_count = 0


    for result in results:

        if result.boxes is None:
            continue


        boxes = result.boxes


        # No tracking IDs available
        if boxes.id is None:
            continue


        track_ids = boxes.id.int().cpu().tolist()

        class_ids = boxes.cls.int().cpu().tolist()

        confidences = boxes.conf.cpu().tolist()

        coordinates = boxes.xyxy.cpu().tolist()


        # ====================================================
        # PROCESS EVERY DETECTED OBJECT
        # ====================================================

        for box, track_id, class_id, confidence in zip(
            coordinates,
            track_ids,
            class_ids,
            confidences
        ):

            x1, y1, x2, y2 = map(
                int,
                box
            )


            # Center
            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )


            # Object name
            class_name = model.names[
                class_id
            ]


            object_count += 1


            # =================================================
            # SAVE POSITION HISTORY
            # =================================================

            position_history[track_id].append(
                (
                    center_x,
                    center_y,
                    current_time
                )
            )


            # =================================================
            # SPEED
            # =================================================

            speed_text = get_speed_text(
                track_id
            )


            # =================================================
            # DRAW OBJECT BOX
            # =================================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # =================================================
            # DRAW CENTER
            # =================================================

            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )


            # =================================================
            # LABEL
            # =================================================

            label = (
                f"{class_name} "
                f"ID:{track_id}"
            )


            cv2.rectangle(
                frame,
                (x1, max(0, y1 - 55)),
                (x1 + 230, y1),
                (0, 0, 0),
                -1
            )


            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )


            # =================================================
            # CONFIDENCE
            # =================================================

            confidence_text = (
                f"Confidence: "
                f"{confidence * 100:.1f}%"
            )


            cv2.putText(
                frame,
                confidence_text,
                (x1 + 5, y1 - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )


            # =================================================
            # SPEED
            # =================================================

            cv2.putText(
                frame,
                f"Speed: {speed_text}",
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )


            # =================================================
            # CENTER COORDINATES
            # =================================================

            position_text = (
                f"Pos: "
                f"{center_x}, {center_y}"
            )


            cv2.putText(
                frame,
                position_text,
                (x1, y2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )


    # ========================================================
    # DASHBOARD
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (300, 100),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        "AI VISION SYSTEM",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"Objects: {object_count}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )


    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        1
    )


    # ========================================================
    # STATUS
    # ========================================================

    cv2.putText(
        frame,
        "AI TRACKING ACTIVE",
        (400, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "AI Multi-Object Vision",
        frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # Quit
    if key == ord("q"):

        break


    # Screenshot
    if key == ord("s"):

        filename = (
            f"ai_detection_"
            f"{int(time.time())}.jpg"
        )

        cv2.imwrite(
            filename,
            frame
        )

        print(
            f"Screenshot saved: {filename}"
        )


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()

print()
print("AI Vision System stopped.")