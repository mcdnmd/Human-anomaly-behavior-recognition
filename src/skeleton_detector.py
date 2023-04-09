import os.path

import cv2
import torch
import numpy as np
from loguru import logger
from torchvision import transforms

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts


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

        skeleton_img, pose_landmarks = self._draw_keypoints(output, image)

        return pose_landmarks, skeleton_img

    def _draw_keypoints(self, output, image, confidence=0.25, threshold=0.65):
        output = non_max_suppression_kpt(
            output,
            confidence,  # Confidence Threshold
            threshold,  # IoU Threshold
            nc=self._pose_model.yaml['nc'],  # Number of Classes
            nkpt=self._pose_model.yaml['nkpt'],  # Number of Keypoints
            kpt_label=True)

        with torch.no_grad():
            output = output_to_keypoint(output)

        nimg = image[0].permute(1, 2, 0) * 255
        nimg = cv2.cvtColor(nimg.cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)

        pose_landmarks = []

        for idx in range(output.shape[0]):
            landmarks, x_y_landmarks = self._extract_landmarks(idx, output)
            plot_skeleton_kpts(nimg, landmarks.T, 3)

        return nimg, pose_landmarks

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

    @staticmethod
    def _extract_landmarks(idx: int, output: np.ndarray):
        landmarks = output[idx, 7:]
        x_y_landmarks = []
        for i in range(0, len(landmarks), 3):
            landmark = landmarks[i:i + 3]
            x_y_landmarks.append((landmark[0], landmark[1]))

        return landmarks, x_y_landmarks