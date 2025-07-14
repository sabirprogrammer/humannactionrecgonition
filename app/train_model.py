# # app/train_model.py

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
import os

def train_model(x_path, y_path, model_dir="model/"):
    # Load data
    X = np.load(x_path)
    y = np.load(y_path)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)

    # Save everything
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "action_model.pkl"))
    joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.pkl"))

    print(" Model trained & saved successfully!")
    
    # Optional: Print Accuracy Report
    preds = model.predict(X)
    print("\n Training Performance:\n")
    print(classification_report(y_encoded, preds, target_names=label_encoder.classes_))
