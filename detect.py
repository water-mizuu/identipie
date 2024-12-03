import cv2
import math
import os
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
from ultralytics import YOLO


# Constants
WINDOW_HEIGHT = 1024
WINDOW_WIDTH = 1280

#Threshold refers to the accepted level of confidence
THRESHOLD = 0.5

# Functions
def update():
    global is_detection_running
    global objects

    ret, img= cap.read()

    if not ret or img is None:
        return
    
    # Resize image to fit Tkinter GUI (width,height)
    img = cv2.resize(img, (int(WINDOW_WIDTH * 0.8), int(WINDOW_HEIGHT * 0.7)))

    if is_detection_running:
        results = model(img, stream=True)

        for r in results:

            # Actual list of detected objects
            boxes = r.boxes
            objects = [(classes[int(box.cls[0])], math.ceil((box.conf[0]*100))/100) for box in boxes]
            # Iterate through each box of object
            for box in boxes:

                # Get model confidence
                confidence = math.ceil((box.conf[0]*100))/100
                
                # Check if object passes treshold
                if (confidence < THRESHOLD):
                    break


                # Get coordinates of bounding box vertices
                x1, y1, x2, y2 = box.xyxy[0]
                # Convert to int
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
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

                cv2.putText(img, f"{classes[box_class]}: {confidence}", org, font, fontScale, color, thickness)    
        # Convert opencv image to RGB for tkinter
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))

        # Update the label with the new image
        screen_container.img = img_tk
        screen_container.config(image=img_tk)

        if is_detection_running:
            root.after(5,update)


def start_detection():
    global is_detection_running
    is_detection_running = True
    update()

def capture_image():
    # Stop live cam
    global is_detection_running
    global objects

    if objects:
        # Pause Live video feed
        is_detection_running = False

        # TODO: Display Nutritional Data
        text_container.config(text=objects[0])

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
model_path = os.path.join('.','models','best.pt')
model = YOLO(model_path)




# Define classes
classes = ["Bread", "Pasta", "Rice", "Apple", "Banana", "BellPepper", "Broccoli", "Chicken", "Fish"]


# Open the video capture with main camera and set resolution to 1920 and 1080
cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

# Create main GUI window
root = tk.Tk()
root.title("Identipie")
root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")

root.bind('<Escape>',lambda e: root.quit())

# Create and place appbar displaying app name
appbar = ttk.Label(root, text="Identipie", font=('Arial', 18))
appbar.grid(row=0,column=0,columnspan=6,padx=8,pady=8)


# Create and place nutrition text container
text_container = ttk.Label(root, text="Sample Text")
text_container.grid(row=1,column=5,columnspan=1,padx=8, pady=8)


# Create and place video camera screen container
screen_container = ttk.Label(root)
screen_container.grid(row=1,column=0,columnspan=4,padx=8,pady=8)

# Create and place 
capture_button = ttk.Button(root, text="Capture Image", command=capture_image)
capture_button.grid(row=2, columnspan=2, padx=8, pady=8)

exit_button = ttk.Button(root, text="Exit Application")
exit_button.grid(row=2, column=2, columnspan=2, padx=8, pady=8)

start_detection()
root.mainloop()

cap.release()
cv2.destroyAllWindows()






