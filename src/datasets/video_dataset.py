import torch
import cv2
from torch.utils.data import Dataset
from PIL import Image


class VideoDatasetUCFCrime(Dataset):
    def __init__(self, video_paths, labels, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, item):
        video_path = self.video_paths[item]
        frames = self.load_video(video_path)

        if self.transform is not None:
            frames = [self.transform(frame) for frame in frames]

        label = self.labels[item]
        return frames, label

    @staticmethod
    def load_video(video_path: str) -> list[Image]:
        capture = cv2.VideoCapture(video_path)
        frames = []
        idx = 1
        while capture.isOpened():
            ret, frame = capture.read()
            if ret:
                frames.append(Image.fromarray(frame))
            else:
                break
            idx += 1
        return frames
