import os
import cv2
import torch
import numpy as np

DATASET_PATH = "shanghaitech/training/frames"
SAVE_DIR = "cached_dataset"

SEQUENCE_LENGTH = 5
IMG_SIZE = 128
CHUNK_SIZE = 2000   # sequences per file

os.makedirs(SAVE_DIR, exist_ok=True)


def save_chunk(data, index):

    data_np = np.array(data, dtype=np.float32)
    tensor = torch.from_numpy(data_np)

    save_path = os.path.join(
        SAVE_DIR,
        f"dataset_part_{index}.pt"
    )

    torch.save(tensor, save_path)

    print(f"✅ Saved chunk {index}")


def build_dataset(folder):

    buffer_seq = []
    chunk_index = 0

    video_folders = sorted(os.listdir(folder))

    for vid in video_folders:

        video_path = os.path.join(folder, vid)

        if not os.path.isdir(video_path):
            continue

        print("Processing:", vid)

        frames = sorted(os.listdir(video_path))
        frame_buffer = []

        for frame_name in frames:

            frame_path = os.path.join(video_path, frame_name)

            img = cv2.imread(frame_path)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))

            frame_buffer.append(img)

            if len(frame_buffer) == SEQUENCE_LENGTH:

                clip = np.concatenate(frame_buffer, axis=0)
                buffer_seq.append(clip)

                frame_buffer.pop(0)

                # ✅ SAVE SMALL CHUNK
                if len(buffer_seq) >= CHUNK_SIZE:
                    save_chunk(buffer_seq, chunk_index)
                    buffer_seq = []
                    chunk_index += 1

    # save remaining
    if buffer_seq:
        save_chunk(buffer_seq, chunk_index)


print("\nBuilding dataset safely...\n")
build_dataset(DATASET_PATH)

print("\n✅ Dataset prepared successfully")