# app/pose_detector.py

import cv2
import mediapipe as mp

class PoseDetector:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(static_image_mode=True)
        self.mp_draw = mp.solutions.drawing_utils

    def extract_keypoints(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return None  # image missing

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.append(lm.x)
                keypoints.append(lm.y)
            return keypoints
        return None  # no pose detected
