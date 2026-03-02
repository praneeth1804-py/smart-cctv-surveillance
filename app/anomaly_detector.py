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
    if fps <= 0:
        fps = 20

    width = 640
    height = 360

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    anomaly_detected = False
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # ✅ PROCESS ONLY EVERY 5TH FRAME
        if frame_count % 5 != 0:
            continue

        # ✅ RESIZE (BIG MEMORY SAVE)
        frame = cv2.resize(frame, (width, height))

        # ✅ LIGHT INFERENCE
        results = model(frame, verbose=False)

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            anomaly_detected = True

        annotated_frame = results[0].plot()

        out.write(annotated_frame)

    cap.release()
    out.release()

    return f"outputs/{output_filename}", anomaly_detected