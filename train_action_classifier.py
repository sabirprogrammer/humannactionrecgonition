# train_action_classifier.py

from app.train_model import train_model

train_model(
    x_path="model/X_train.npy",
    y_path="model/y_train.npy",
    model_dir="model/"
)
