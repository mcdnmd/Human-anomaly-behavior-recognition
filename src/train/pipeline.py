import os
import cv2
import pandas as pd
from pathlib import Path

from src import config
from src.datasets.video_dataset import VideoDatasetUCFCrime
from src.skeleton_detector import SkeletonDetector


def create_ufc_crime_dataset(dataset_folder: str) -> VideoDatasetUCFCrime:
    path = Path(dataset_folder)

    video_paths = []
    for folder in os.listdir(path):
        for video in os.listdir(path / str(folder)):
            video_path = path / str(folder) / str(video)
            video_paths.append(str(video_path))

    df = pd.DataFrame()
    df["video_paths"] = video_paths
    df["label"] = df["video_paths"].apply(lambda x: os.path.basename(x).split("_")[0][:-3].lower())

    return VideoDatasetUCFCrime(
        video_paths=df["video_paths"].tolist(),
        labels=df['label'].tolist()
    )


if __name__ == '__main__':
    dataset = create_ufc_crime_dataset("/Users/kir/Downloads/Anomaly-Videos-Part-1")
    skeleton_detector = SkeletonDetector(config.ROOT / 'data/yolov7-w6-pose.pt')

    frames, label = dataset[50]
    for frame in frames:
        pose, img = skeleton_detector.predict(frame)
        if pose:
            print(frame, pose)
