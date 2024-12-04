import cv2
import math
import os
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
from ultralytics import YOLO
import json


# Constants
WINDOW_HEIGHT = 968
WINDOW_WIDTH = 1526

# Threshold refers to the accepted level of confidence
THRESHOLD = 0.5
LOW_THRESHOLD = 0.8

# TKInter constants
TITLE_FONT = ('Arial', 20)
SUBTITLE_FONT = ('Arial', 18)
SUBTEXT_FONT = ('Arial', 16)

# Functions
def update():
    global is_detection_running
    global objects
    global confidence

    ret, img = cap.read()

    if not ret or img is None:
        return
    
    # Resize image to fit Tkinter GUI (width, height)
    img = cv2.resize(img, (int(WINDOW_WIDTH * 0.65), int(WINDOW_HEIGHT * 0.7)))

    if is_detection_running:
        for r in model(img, stream=True):
            # Actual list of detected objects
            boxes = r.boxes
            objects = [(classes[int(box.cls[0])], math.ceil((box.conf[0]*100))/100)
                       for box in boxes]
            # Iterate through each box of object
            for box in boxes:
                # Get model confidence
                measured_confidence = math.ceil((box.conf[0]*100))/100
                
                # Check if object passes treshold
                if measured_confidence < THRESHOLD:
                    break

                confidence = measured_confidence
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

    warning_label.config(text="")

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
        info = definitions[objects[0][0].lower()]

        text_container.config(text=f"{info['name']} Detected")
        per_serving_label.config(text=f"Each serving includes:")
        
        calorie_label.config(text=f"Calories")
        calorie_container.config(text=f"{info['calories']}cal")
        
        fat_label.config(text=f"Fat")
        fat_container.config(text=f"{info['fat']}g")
        
        carbohydrates_label.config(text=f"Carbohydrates")
        carbohydrates_container.config(text=f"{info['carbohydrates']}g")
        
        protein_label.config(text=f"Protein")
        protein_container.config(text=f"{info['protein']}g")
        
        if confidence < LOW_THRESHOLD:
            warning_label.config(text=f"Warning: The confidence {confidence * 100.00}% is relatively low. \nThis may result in inaccuracies of the measurements.")

        # TODO: Change Button to Capture Another Picture
        capture_button.config(text="Capture Another Image", command=start_detection)
    else:
        text_container.config(text="No food object found")


# global variables
is_detection_running = True
objects = list()
confidence = 0.0

# Load Pretrained model
model_path = os.path.join('.', 'models', 'best.pt')
model = YOLO(model_path)

# Define classes
classes = ["Bread", "Pasta", "Rice", "Apple", "Banana", "BellPepper", "Broccoli", "Chicken", "Fish"]

# Read the definitions from the json file
with open('definitions.json') as f:
    definitions = json.load(f)

# Open the video capture with main camera and set resolution to 1920 and 1080
cap = cv2.VideoCapture(0)
cap.set(3, 1600)
cap.set(4, 900)

# Create main GUI window
root = tk.Tk()
root.title("Identipie")
root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")

root.bind('<Escape>', lambda _: root.quit())
# root.bind('<Return>', lambda _: capture_image())
# root.bind('<space>', lambda _: capture_image())

# Create and place appbar displaying app name
appbar = ttk.Label(root, text="Identipie", font=TITLE_FONT)
appbar.grid(row=0, column=0, columnspan=6, padx=8, pady=8)

# Create and place nutrition text container
text_container = ttk.Label(root, text="Capture Image", font=TITLE_FONT)
text_container.grid(row=1, column=5, columnspan=1, padx=8, pady=8)

per_serving_label = ttk.Label(root, text="Each serving includes:", font=SUBTITLE_FONT, anchor='w')
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

warning_label = ttk.Label(root, text="Tete", font=SUBTEXT_FONT)
warning_label.grid(row=9, column=0, columnspan=2, rowspan=6, padx=8, pady=4, sticky='w')

# Create and place video camera screen container
screen_container = ttk.Label(root)
screen_container.grid(row=1, column=0, columnspan=4, rowspan=6, padx=8, pady=8)
 
# Create and place 
capture_button = ttk.Button(root, text="Capture ", command=capture_image)
capture_button.grid(row=8, columnspan=2, padx=8, pady=8)

exit_button = ttk.Button(root, text="Exit Application", command=lambda: root.quit())
exit_button.grid(row=8, column=2, columnspan=2, padx=8, pady=8)

start_detection()
root.mainloop()

cap.release()
cv2.destroyAllWindows()






