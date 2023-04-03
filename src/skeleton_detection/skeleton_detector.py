import os.path

import torch
import numpy as np
from loguru import logger
from torchvision import transforms

from yolov7.utils.datasets import letterbox


class SkeletonDetector:
    def __init__(self, weights: str):

        self._cuda_is_available = torch.cuda.is_available()
        if not self._cuda_is_available:
            logger.warning("GPU is not available")
            self._device = torch.device("cpu")
        else:
            self._device = torch.device("cuda:0")

        if not os.path.exists(weights):
            logger.exception(f"Weights {weights} not exists")
            raise ValueError
        self._weights = torch.load(weights, map_location=self._device)
        self._pose_model = self._weights['model']
        self._pose_model.float().eval()

        if self._cuda_is_available:
            self._pose_model.half().to(self._device)

    def predict(self, frame: np.array) -> (list[float], np.array):
        """Predict a keypoint of skeleton"""
        image = letterbox(frame, 960, stride=64, auto=True)[0]
        image = transforms.ToTensor()(image)

        if self._cuda_is_available:
            image = image.half().to(self._device)

        image = image.unsqueeze(0)

        with torch.no_grad():
            output, _ = self._pose_model(image)

        return output, image

    @staticmethod
    def normalize_skeletons(skeletons: list[np.array]) -> list[np.array]:
        """Skeleton normalization"""
        normalized_skeletons = []
        for skeleton in skeletons:
            coords = []
            for i in range(0, len(skeleton), 3):
                landmark = skeleton[i:i + 3]
                coords.append([landmark[0], landmark[1]])
            coords = np.array(coords)
            center = np.mean([coords[1], coords[2], coords[5], coords[8], coords[11]], axis=0)
            max_dist = np.max(np.linalg.norm(coords - center, axis=1))
            normalized_skeleton = (coords - center) / max_dist
            normalized_skeletons.append(normalized_skeleton)

        return normalized_skeletons
