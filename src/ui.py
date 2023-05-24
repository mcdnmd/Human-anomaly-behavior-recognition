import os

import gradio as gr
import cv2

from loguru import logger
from src import config

from src.skeleton_detector import SkeletonDetector


detector = SkeletonDetector(config.ROOT / 'data/yolov7-w6-pose.pt')


def detect_image(inpt_img):
    img = cv2.imread(inpt_img)
    output, image = detector.predict(img)
    return image


def detect_video(inpt_video):
    capture = cv2.VideoCapture(inpt_video)

    video_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    video_fps = capture.get(cv2.CAP_PROP_FPS)

    landmark_poses = []

    idx = 1
    fps = 1

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    output_video_path = config.ROOT / f"data/out_{os.path.basename(inpt_video)}"
    out_video = cv2.VideoWriter(str(output_video_path.absolute()), fourcc, fps, (int(capture.get(3)), int(capture.get(4))))

    while capture.isOpened():
        ret, frame = capture.read()

        if ret:
            # if idx % fps == 1:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output, image = detector.predict(image)
            # Todo: add anomaly detector
            frame = cv2.resize(image, (int(capture.get(3)), int(capture.get(4))))
            out_video.write(frame)
            landmark_poses.append(output)
        else:
            break
        idx += 1
    capture.release()
    out_video.release()
    logger.info("Complite %s" % str(output_video_path))
    print(landmark_poses)
    return str(output_video_path)


def create_web_app():
    images = gr.Interface(detect_image, gr.Image(type='filepath'), "image")
    videos = gr.Interface(detect_video, gr.Video(type='filepath'), "video")
    gr.TabbedInterface(
        [images, videos], ['Images', 'Video']
    ).launch()
