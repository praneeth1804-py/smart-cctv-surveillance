import os
import cv2
import time
import subprocess
import math
from ultralytics import YOLO
from collections import defaultdict, deque


# ===============================
# LOAD YOLO
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "app", "yolov8n.pt")
model = YOLO(MODEL_PATH)

# ===============================
# TRACK MEMORY
# ===============================
track_history = defaultdict(lambda: deque(maxlen=8))

# movement multiplier
MOVEMENT_RATIO = 1.2


def process_video(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Video open failed")

    filename = os.path.splitext(os.path.basename(video_path))[0]
    unique_id = int(time.time())

    output_dir = os.path.join(BASE_DIR, "static", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    temp_video = os.path.join(output_dir, f"temp_{unique_id}.avi")
    final_video = os.path.join(output_dir, f"processed_{filename}_{unique_id}.mp4")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1 or fps > 120:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        temp_video,
        cv2.VideoWriter_fourcc(*'XVID'),
        fps,
        (width, height)
    )

    anomaly_found = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, classes=[0])

        if results[0].boxes.id is not None:

            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()

            for box, track_id in zip(boxes, ids):

                x1, y1, x2, y2 = map(int, box)

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                box_width = max((x2 - x1), 1)

                history = track_history[track_id]
                history.append((cx, cy))

                suspicious = False

                # compare long displacement
                if len(history) >= 6:
                    old_x, old_y = history[0]

                    movement = math.dist(
                        (cx, cy),
                        (old_x, old_y)
                    )

                    # movement relative to body size
                    if movement > box_width * MOVEMENT_RATIO:
                        suspicious = True

                if suspicious:
                    color = (0, 0, 255)
                    label = "SUSPICIOUS"
                    anomaly_found = True
                else:
                    color = (0, 255, 0)
                    label = "Normal"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        out.write(frame)

    cap.release()
    out.release()

    subprocess.run([
        r"C:\ffmpeg-2026-02-23-git-7b15039cdb-essentials_build\bin\ffmpeg.exe",
        "-y",
        "-i", temp_video,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        final_video
    ], check=True)

    os.remove(temp_video)

    return (
        "outputs/" + os.path.basename(final_video),
        anomaly_found
    )