from flask import Flask, render_template, request
import os
from anomaly_detector import process_video


# =====================================
# PROJECT ROOT
# =====================================
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")


# =====================================
# FLASK INIT
# =====================================
app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)


# =====================================
# FOLDERS
# =====================================
UPLOAD_FOLDER = os.path.join(STATIC_DIR, "uploads")
SAMPLE_FOLDER = os.path.join(STATIC_DIR, "samples")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAMPLE_FOLDER, exist_ok=True)


# =====================================
# HOME PAGE
# =====================================
@app.route("/")
def home():
    return render_template("index.html")


# =====================================
# VIDEO UPLOAD
# =====================================
@app.route("/upload", methods=["POST"])
def upload():

    if "video" not in request.files:
        return "No video uploaded"

    file = request.files["video"]

    if file.filename == "":
        return "Empty filename"

    input_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(input_path)

    output_video, anomaly_flag = process_video(input_path)

    return render_template(
        "result.html",
        video=output_video,
        anomaly=anomaly_flag
    )


# =====================================
# SAMPLE VIDEO PAGE
# =====================================
@app.route("/samples")
def samples():

    videos = os.listdir(SAMPLE_FOLDER)

    return render_template(
        "samples.html",
        videos=videos
    )


# =====================================
# RUN SAMPLE VIDEO
# =====================================
@app.route("/run_sample/<video>")
def run_sample(video):

    sample_path = os.path.join(
        SAMPLE_FOLDER,
        video
    )

    output_video, anomaly_flag = process_video(sample_path)

    return render_template(
        "result.html",
        video=output_video,
        anomaly=anomaly_flag
    )


# =====================================
# RUN SERVER
# =====================================
if __name__ == "__main__":
    app.run()