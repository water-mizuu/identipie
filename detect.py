import cv2
import math
import os
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
from ultralytics import YOLO



# Load Pretrained model
model_path = os.path.join('.','models','best.pt')
model = YOLO(model_path)

#Treshold refers to the accepted level of confidence
treshold = 0.5

# Define classes
classes = ["Bread", "Pasta", "Rice", "Apple", "Banana", "BellPepper", "Broccoli", "Chicken", "Fish"]


# Open the video capture with main camera
cap = cv2.VideoCapture(0)

# Set camera resolution
cap.set(3, 1920)
cap.set(4, 1080)


# Create main GUI window
root = tk.Tk()
root.title("Food Object Detection")

# root.bind('<Escape>',lambda e: app.quit())

# Create a label to display the video stream
label = ttk.Label(root)
label.pack(padx=16, pady=16)

start_button = ttk.Button(root, text="Capture Image")
start_button.pack(side=tk.LEFT, padx=8)

exit_button = ttk.Button(root, text="Exit Application")
exit_button.pack(side=tk.LEFT,padx=8)


def on_key_press(event):
    # Check if event 
    if event.char == 'q':
        root.destroy()

def update():
    ret, img= cap.read()

    if not ret or img is None:
        return
    

    results = model(img, stream=True)
    for r in results:

        # Actual list of detected objects
        boxes = r.boxes
        # Iterate through each box of object
        for box in boxes:

            # Get model confidence
            confidence = math.ceil((box.conf[0]*100))/100
            
            # Check if object passes treshold
            if (confidence < 0.2):
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
    label.img = img_tk
    label.config(image=img_tk)

    root.after(5,update)



update()
root.mainloop()

cap.release()
cv2.destroyAllWindows()





