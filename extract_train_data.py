# extract_train_data.py

from app.data_loader import create_dataset

create_dataset(
    image_folder="data/train",
    excel_file="data/training_set.csv",
    output_x="model/X_train.npy",
    output_y="model/y_train.npy"
)
