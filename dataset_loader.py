import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np


class VideoDataset(Dataset):

    def __init__(self, frame_path):
        self.samples = []

        for folder in os.listdir(frame_path):
            folder_path = os.path.join(frame_path, folder)

            frames = sorted(os.listdir(folder_path))

            for i in range(len(frames) - 4):
                clip = frames[i:i+5]
                clip_paths = [
                    os.path.join(folder_path, f)
                    for f in clip
                ]
                self.samples.append(clip_paths)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        clip_paths = self.samples[idx]
        frames = []

        for path in clip_paths:
            img = cv2.imread(path)
            img = cv2.resize(img, (128,128))
            img = img / 255.0
            img = np.transpose(img, (2,0,1))
            frames.append(img)

        frames = np.concatenate(frames, axis=0)

        return torch.tensor(frames, dtype=torch.float32)