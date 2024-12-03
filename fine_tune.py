# Load necessary libraries
from ultralytics import YOLO
import os
import sys



# Example data/set 1/config.yaml
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path of config file>")
    else:
        path = sys.argv[1]

        yolo_model = YOLO('models/best.pt') 
        results = yolo_model.train(data=path, epochs=5, batch=8, resume=True, device="cpu")
        metrics = yolo_model.val(data=path)
        
        yolo_model.export(format="onnx")
