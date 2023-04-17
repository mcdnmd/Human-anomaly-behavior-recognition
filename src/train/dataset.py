import os
import pandas as pd
from pathlib import Path

from src.datasets.video_dataset import VideoDatasetUCFCrime

if __name__ == '__main__':
    path = Path("/Users/kir/Downloads/Anomaly-Videos-Part-1")

    video_paths = []
    for folder in os.listdir(path):
        for video in os.listdir(path / str(folder)):
            video_path = path / str(folder) / str(video)
            video_paths.append(str(video_path))

    df = pd.DataFrame()
    df["video_paths"] = video_paths
    df["label"] = df["video_paths"].apply(lambda x: os.path.basename(x).split("_")[0][:-3].lower())

    dataset = VideoDatasetUCFCrime(
        video_paths=df["video_paths"].tolist(),
        labels=df['label'].tolist()
    )
    print(dataset)
    print(dataset[2])
