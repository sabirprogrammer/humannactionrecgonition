# app/data_loader.py

import os
import pandas as pd
from app.pose_detector import PoseDetector
from tqdm import tqdm
import numpy as np

def create_dataset(image_folder, excel_file, output_x='X.npy', output_y='y.npy'):
    df = pd.read_csv(excel_file)
    detector = PoseDetector()

    X, y = [], []

    print("Extracting pose keypoints...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(image_folder, row['filename'])
        label = row['label']

        keypoints = detector.extract_keypoints(img_path)
        if keypoints:
            X.append(keypoints)
            y.append(label)

    print(f" Done! Total samples with pose: {len(X)}")
    np.save(output_x, np.array(X))
    np.save(output_y, np.array(y))
