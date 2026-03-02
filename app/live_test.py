import cv2
import yt_dlp
from app.anomaly_detector import model, track_history
import math


# ===============================
# GET YOUTUBE STREAM
# ===============================
def get_stream_url(youtube_url):

    ydl_opts = {"format": "best"}

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info["url"]


# ===============================
# LIVE DETECTION
# ===============================
def live_detection(youtube_url):

    stream_url = get_stream_url(youtube_url)

    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("Stream failed")
        return

    print("LIVE STARTED — press Q to quit")

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

                cx = int((x1 + x2)/2)
                cy = int((y1 + y2)/2)

                history = track_history[track_id]
                history.append((cx, cy))

                suspicious = False

                if len(history) >= 6:

                    old_x, old_y = history[0]

                    movement = math.dist(
                        (cx, cy),
                        (old_x, old_y)
                    )

                    width = max((x2-x1), 1)

                    if movement > width * 1.2:
                        suspicious = True

                if suspicious:
                    color = (0,0,255)
                    label = "SUSPICIOUS"
                else:
                    color = (0,255,0)
                    label = "Normal"

                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                cv2.putText(
                    frame,label,
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,color,2
                )

        cv2.imshow("LIVE CCTV DETECTION", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()