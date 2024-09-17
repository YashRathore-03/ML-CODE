import cv2
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
from roboflow import Roboflow

# Initialize Roboflow
rf = Roboflow(api_key="H9KUpRRCRNN9AbiPFVrL")
project = rf.workspace().project("final-weed-detection-7zq5o")
model = project.version(1).model

# Initialize Tkinter window
root = tk.Tk()
root.title("Weed Detection")

# Create a label to show the video feed
label = Label(root)
label.pack()

# Create a button to start the camera feed
def start_camera():
    global cap, is_camera_on
    if not is_camera_on:
        cap = cv2.VideoCapture(0)
        is_camera_on = True
        update_frame()

def detect_weed(frame):
    # Save the frame to a temporary file
    temp_file = "temp_frame.jpg"
    cv2.imwrite(temp_file, frame)

    # Perform inference
    predictions = model.predict(temp_file, confidence=40, overlap=30).json()

    # Draw bounding boxes on the frame
    for pred in predictions['predictions']:
        x1, y1, x2, y2 = int(pred['x'] - pred['width'] / 2), int(pred['y'] - pred['height'] / 2), int(pred['x'] + pred['width'] / 2), int(pred['y'] + pred['height'] / 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return frame

def update_frame():
    global cap
    # Capture frame from camera
    ret, frame = cap.read()

    if ret:
        # Detect weed and draw bounding boxes
        frame = detect_weed(frame)

        # Convert the frame to an ImageTk object
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)

        # Update the Tkinter label
        label.img_tk = img_tk
        label.configure(image=img_tk)

    # Repeat this function after a delay
    root.after(10, update_frame)

# Create a button to start the camera feed
is_camera_on = False
start_button = Button(root, text="Start Camera", command=start_camera)
start_button.pack()

# Start the Tkinter event loop
root.mainloop()

# Release the camera when done
if is_camera_on:
    cap.release()