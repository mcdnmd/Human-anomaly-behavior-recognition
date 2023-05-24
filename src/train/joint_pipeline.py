import os
from pathlib import Path

import torch.utils

from src.datasets.joints_dataset import JointDatasetUCFCrime


def create_joint_dataset(dataset_folder):
    path = Path(dataset_folder)
    files = []
    for file in os.listdir(path):
        files.append(str(path / file))
    dataset = JointDatasetUCFCrime(files)

    return dataset


if __name__ == '__main__':
    dataset = create_joint_dataset('/Users/kir/Downloads/skeleton-anomaly')
    train_set, val_set = torch.utils.data.random_split(dataset, [700, 300])
    print(train_set)
    print(val_set)
