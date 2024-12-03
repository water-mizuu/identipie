import cv2
import math
import os
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
from ultralytics import YOLO
import json


# Constants
WINDOW_HEIGHT = 1024
WINDOW_WIDTH = 1280

# Threshold refers to the accepted level of confidence
THRESHOLD = 0.5

# TKInter constants
TITLE_FONT = ('Arial', 20)
SUBTITLE_FONT = ('Arial', 18)
SUBTEXT_FONT = ('Arial', 16)

# MACRO_NUTRIENTS
#   Extracted from: https://developer.edamam.com/edamam-nutrition-api-demo
DEFINITIONS = {
  "bread": {
    "name": "Bread",
    "calories": 164,
    "fat": 2.7,
    "carbohydrates": 28.5,
    "protein": 6.4
  },
  "pasta": {
    "name": "Pasta",
    "calories": 1113,
    "fat": 4.5,
    "carbohydrates": 224.1,
    "protein": 39
  },
  "apple": {
    "name": "Apple",
    "calories": 126,
    "fat": 0.4,
    "carbohydrates": 33.4,
    "protein": 0.6
  },
  "banana": {
    "name": "Banana",
    "calories": 112,
    "fat": 0.4,
    "carbohydrates": 28.7,
    "protein": 1.4
  },
  "bellpepper": {
    "name": "Bell Pepper",
    "calories": 3,
    "fat": 0.0,
    "carbohydrates": 0.6,
    "protein": 0.1
  },
  "broccoli": {
    "name": "Broccoli",
    "calories": 50,
    "fat": 0.5,
    "carbohydrates": 9.8,
    "protein": 4.2
  },
  "chicken": {
    "name": "Chicken",
    "calories": 538,
    "fat": 37.8,
    "carbohydrates": 0,
    "protein": 46.0
  },
  "fish": {
    "name": "Fish",
    "calories": 111,
    "fat": 2.0,
    "carbohydrates": 0,
    "protein": 20.0
  }
}

# Functions
def update():
    global is_detection_running
    global objects

    ret, img = cap.read()

    if not ret or img is None:
        return
    
    # Resize image to fit Tkinter GUI (width, height)
    img = cv2.resize(img, (int(WINDOW_WIDTH * 0.8), int(WINDOW_HEIGHT * 0.7)))

    if is_detection_running:
        results = model(img, stream=True)

        for r in results:

            # Actual list of detected objects
            boxes = r.boxes
            objects = [(classes[int(box.cls[0])], math.ceil((box.conf[0]*100))/100)
                       for box in boxes]
            # Iterate through each box of object
            for box in boxes:
                # Get model confidence
                confidence = math.ceil((box.conf[0]*100))/100
                
                # Check if object passes treshold
                if confidence < THRESHOLD:
                    break

                # Get coordinates of bounding box vertices
                # Convert to int
                x1, y1, x2, y2 = [int(x) for x in box.xyxy[0]]
                # Outline box in image with rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Get object class index
                box_class = int(box.cls[0])

                # Rectangle Additional Details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(
                    img, 
                    f"{classes[box_class]}: {confidence}", 
                    org, 
                    font, 
                    fontScale, 
                    color, 
                    thickness)

        # Convert opencv image to RGB for tkinter
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))

        # Update the label with the new image
        screen_container.img = img_tk
        screen_container.config(image=img_tk)

        if is_detection_running:
            root.after(5, update)


def start_detection():
    global is_detection_running
    is_detection_running = True

    text_container.config(text=f"Capture Image")
    per_serving_label.config(text=f"")
    
    calorie_label.config(text="")
    calorie_container.config(text="")

    fat_label.config(text="")
    fat_container.config(text="")

    carbohydrates_label.config(text="")
    carbohydrates_container.config(text="")

    protein_label.config(text="")
    protein_container.config(text="")

    capture_button.config(text="Capture Image", command=capture_image)

    update()

def capture_image():
    # Stop live cam
    global is_detection_running
    global objects

    if objects:
        # Pause Live video feed
        is_detection_running = False

        # TODO: Display Nutritional Data
        info = DEFINITIONS[objects[0][0].lower()]

        text_container.config(text=f"{info['name']} Detected")
        per_serving_label.config(text=f"Per serving...")
        
        calorie_label.config(text=f"Calories")
        calorie_container.config(text=f"{info['calories']}cal")
        
        fat_label.config(text=f"Fat")
        fat_container.config(text=f"{info['fat']}g")
        
        carbohydrates_label.config(text=f"Carbohydrates")
        carbohydrates_container.config(text=f"{info['carbohydrates']}g")
        
        protein_label.config(text=f"Protein")
        protein_container.config(text=f"{info['protein']}g")

        # TODO: Change Button to Capture Another Picture
        capture_button.config(text="Capture Another Image", command=start_detection)
    else:
        text_container.config(text="No food object found")


def on_key_press(event):
    # Check if event 
    if event.char == 'q':
        root.destroy()


# global variables
is_detection_running = True
objects = list()

# Load Pretrained model
model_path = os.path.join('.', 'models', 'best.pt')
model = YOLO(model_path)

# Define classes
classes = ["Bread", "Pasta", "Rice", "Apple", "Banana", "BellPepper", "Broccoli", "Chicken", "Fish"]

# Open the video capture with main camera and set resolution to 1920 and 1080
cap = cv2.VideoCapture(0)
cap.set(3, 1600)
cap.set(4, 900)

# Create main GUI window
root = tk.Tk()
root.title("Identipie")
root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")

root.bind('<Escape>', lambda _: root.quit())

# Create and place appbar displaying app name
appbar = ttk.Label(root, text="Identipie", font=TITLE_FONT)
appbar.grid(row=0, column=0, columnspan=6, padx=8, pady=8)

# Create and place nutrition text container
text_container = ttk.Label(root, text="Capture Image", font=TITLE_FONT)
text_container.grid(row=1, column=5, columnspan=1, padx=8, pady=8)

per_serving_label = ttk.Label(root, text="Per serving...", font=SUBTITLE_FONT, anchor='w')
per_serving_label.grid(row=2, column=5, columnspan=2, padx=8, pady=0, sticky='w')

calorie_label = ttk.Label(root, text="Calories", font=SUBTITLE_FONT, anchor='w')
calorie_label.grid(row=3, column=5, columnspan=1, padx=8, pady=4, sticky='w')

calorie_container = ttk.Label(root, text="Calories", font=SUBTITLE_FONT, anchor='e')
calorie_container.grid(row=3, column=6, columnspan=1, padx=8, pady=4, sticky='e')

fat_label = ttk.Label(root, text="Fats", font=SUBTEXT_FONT, anchor='w')
fat_label.grid(row=4, column=5, columnspan=1, padx=8, pady=4, sticky='w')

fat_container = ttk.Label(root, text="Fats", font=SUBTEXT_FONT, anchor='e')
fat_container.grid(row=4, column=6, columnspan=1, padx=8, pady=4, sticky='e')

carbohydrates_label = ttk.Label(root, text="Carbohydrates", font=SUBTEXT_FONT, anchor='w')
carbohydrates_label.grid(row=5, column=5, columnspan=1, padx=8, pady=4, sticky='w')

carbohydrates_container = ttk.Label(root, text="Carbohydrates", font=SUBTEXT_FONT, anchor='e')
carbohydrates_container.grid(row=5, column=6, columnspan=1, padx=8, pady=4, sticky='e')

protein_label = ttk.Label(root, text="Proteins", font=SUBTEXT_FONT, anchor='w')
protein_label.grid(row=6, column=5, columnspan=1, padx=8, pady=4, sticky='w')

protein_container = ttk.Label(root, text="Proteins", font=SUBTEXT_FONT, anchor='e')
protein_container.grid(row=6, column=6, columnspan=1, padx=8, pady=4, sticky='e')

# Create and place video camera screen container
screen_container = ttk.Label(root)
screen_container.grid(row=1, column=0, columnspan=4, rowspan=6, padx=8, pady=8)
 
# Create and place 
capture_button = ttk.Button(root, text="Capture ", command=capture_image)
capture_button.grid(row=7, columnspan=2, padx=8, pady=8)

exit_button = ttk.Button(root, text="Exit Application", command=lambda: root.quit())
exit_button.grid(row=7, column=2, columnspan=2, padx=8, pady=8)

start_detection()
root.mainloop()

cap.release()
cv2.destroyAllWindows()






