import pandas as pd
from torch.utils.data import Dataset


class JointDatasetUCFCrime(Dataset):
    def __init__(self, data_paths: list[str], transform=None):
        self.data_paths = data_paths
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, item):
        data_path = self.data_paths[item]
        df = pd.read_csv(data_path, sep=';')
        return df
