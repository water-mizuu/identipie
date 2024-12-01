# Load necessary libraries
from ultralytics import YOLO

# Build new model from scratch
yolo_model = YOLO('yolov8n.yaml') 

# 
results = yolo_model.train(data="config.yaml", epochs=20, batch=8, resume=True, device="cpu")
metrics = yolo_model.val(data="config.yaml")

yolo_model.export(format="onnx")