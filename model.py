# NOTEBOOK SOURCE: https://www.kaggle.com/code/watermizuu/human-faces-object-detection-yolov8/edit

# Importing necessary libraries
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
import torch
import subprocess
from dotenv import dotenv_values

from ultralytics import YOLO

# Import wandb and log in
import wandb

config = dotenv_values(".env")

# Log in to wandb with API key
wandb.login(key=config["key"])

# File paths
image_dir = "./face_detection/images"
csv_path = "./face_detection/faces.csv"
output_dir = "./output"

# Load the dataset
faces_df = pd.read_csv(csv_path)

# EDA
print(faces_df.head())
print(faces_df.info())
print(faces_df.describe())

# Convert bounding box coordinates to YOLO format (normalized x_center, y_center, width, height)
def convert_to_yolo(row):
    dw = 1. / row['width']
    dh = 1. / row['height']
    x = (row['x0'] + row['x1']) / 2.0
    y = (row['y0'] + row['y1']) / 2.0
    w = row['x1'] - row['x0']
    h = row['y1'] - row['y0']
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


print("Converting bounding boxes to YOLO format")
faces_df[['x', 'y', 'w', 'h']] = faces_df.apply(convert_to_yolo, axis=1, result_type='expand')

print("Splitting dataset into train and validation sets")
# Split the dataset into train and validation sets
train_df, val_df = train_test_split(faces_df, test_size=0.2, random_state=42)

print("Creating annotation files")
# Create YOLO annotation files
annotations_dir = os.path.join(output_dir, 'annotations')
os.makedirs(annotations_dir, exist_ok=True)

for idx, row in faces_df.iterrows():
    annotation_path = os.path.join(annotations_dir, row['image_name'].replace('.jpg', '.txt'))
    with open(annotation_path, 'w') as f:
        f.write(f"0 {row['x']} {row['y']} {row['w']} {row['h']}\n")
        
# Function to create YOLO annotation files
def create_annotations(df, annotations_dir):
    os.makedirs(annotations_dir, exist_ok=True)
    for idx, row in df.iterrows():
        annotation_path = os.path.join(annotations_dir, row['image_name'].replace('.jpg', '.txt'))
        with open(annotation_path, 'w') as f:
            f.write(f"0 {row['x']} {row['y']} {row['w']} {row['h']}\n")

print("Creating YOLO annotation files for train and validation sets")
# Create YOLO annotation files for train and validation sets
train_annotations_dir = os.path.join(output_dir, 'train', 'labels')
create_annotations(train_df, train_annotations_dir)

val_annotations_dir = os.path.join(output_dir, 'val', 'labels')
create_annotations(val_df, val_annotations_dir)

# Image preprocessing
def preprocess_images(image_list, source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for img_name in image_list:
        img_path = os.path.join(source_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            target_path = os.path.join(target_dir, img_name)
            cv2.imwrite(target_path, img)

print("Preprocessing images for train and validation sets")
train_images_dir = os.path.join(output_dir, 'train', 'images')
preprocess_images(train_df['image_name'].tolist(), image_dir, train_images_dir)

val_images_dir = os.path.join(output_dir, 'val', 'images')
preprocess_images(val_df['image_name'].tolist(), image_dir, val_images_dir)

# Verify annotation files
print("Train annotations sample:")
print(os.listdir(train_annotations_dir)[:5])

print("Validation annotations sample:")
print(os.listdir(val_annotations_dir)[:5])

yolo_model = YOLO('yolov8n.yaml') 

# Define the data.yaml file content dynamically
class_names = ['face']
data_yaml_content = f"""
train: {os.path.abspath(train_images_dir)}
val: {os.path.abspath(val_images_dir)}
nc: {len(class_names)}
names: {class_names}
"""

print("Saving data.yaml file")
# Save the data.yaml file
data_yaml_path = os.path.join(output_dir, 'data.yaml')
with open(data_yaml_path, 'w') as f:
    f.write(data_yaml_content)

# Train the model
yolo_model.train(data=data_yaml_path, epochs=50, imgsz=640)

# Function to draw rounded rectangle
def draw_rounded_rectangle(img, start_point, end_point, color, thickness, radius):
    x1, y1 = start_point
    x2, y2 = end_point
    img = cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    img = cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    img = cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    img = cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    img = cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    img = cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)
    return img

# Display 10 sample images with detection results
def display_samples_with_detections(model, images_dir, samples=10):
    sample_images = os.listdir(images_dir)[:samples]
    for img_name in sample_images:
        img_path = os.path.join(images_dir, img_name)
        results = model(img_path)[0]  # Get the first (and only) Results object
        img = cv2.imread(img_path)
        for box in results.boxes.xyxy.cpu().numpy():  # x1, y1, x2, y2
            x1, y1, x2, y2 = map(int, box)
            img = draw_rounded_rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2, 10)  # Red color, thickness 2, radius 10
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

display_samples_with_detections(yolo_model, val_images_dir)

print("Process completed. Outputs are stored in:", output_dir)