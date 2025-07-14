# # realtime_predict.py

import cv2
import numpy as np
import mediapipe as mp
import joblib

# Load trained model and encoder
model = joblib.load("model/action_model.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

# Init MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract 33 landmarks
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.append(lm.x)
            keypoints.append(lm.y)

        if len(keypoints) == 66:
            keypoints_np = np.array(keypoints).reshape(1, -1)
            prediction = model.predict(keypoints_np)
            predicted_label = label_encoder.inverse_transform(prediction)[0]

            # Show on screen
            cv2.putText(frame, f'Action: {predicted_label}', (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Pose Action Recognition", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
