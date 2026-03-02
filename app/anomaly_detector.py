import os
import cv2
from ultralytics import YOLO

# =====================================
# LOAD MODEL ONCE (GLOBAL)
# =====================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

model = YOLO(MODEL_PATH)

# =====================================
# PROCESS VIDEO
# =====================================
def process_video(input_path):

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise Exception("Could not open video file")

    output_filename = "processed_" + os.path.basename(input_path)
    output_path = os.path.join(
        BASE_DIR,
        "static",
        "outputs",
        output_filename
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = 640
    height = 360

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    anomaly_detected = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize to reduce memory usage
        frame = cv2.resize(frame, (width, height))

        # Run detection (NO tracking)
        results = model(frame)

        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            anomaly_detected = True

        annotated_frame = results[0].plot()

        out.write(annotated_frame)

    cap.release()
    out.release()

    return f"outputs/{output_filename}", anomaly_detected