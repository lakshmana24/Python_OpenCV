**Python OpenCV Projects: SpeakSight & Depth Detection**

This repository contains three Python projects leveraging AI and computer vision to assist with image/object description and depth estimation. The solutions are designed to be beneficial for accessibility, robotics, and general computer vision experimentation.

**Project List**
 * SpeakSight.py
 * depth.py
 * distance.py

**SpeakSight.py**

Description:
  SpeakSight is an AI-powered tool that helps visually impaired users by detecting objects in images and generating spoken descriptions.

  Accepts an input image, analyzes it, and generates a descriptive summary.
  The description is delivered as speech for accessibility.
  
**Tech Stack:**
  Python
  YOLOv8 (Ultralytics)
  Tkinter
  Pillow (PIL)
  pyttsx3
  random
  NumPy
    
**DEPTH DETECTION**

DEPTH DETECTION is the combined system comprising depth.py and distance.py.

depth.py
  Calculates the depth of an object in real time using the webcam feed.
  Utilizes computer vision to estimate how far objects are from the camera.

distance.py
  Detects objects using YOLOv8 and calculates their distance from the webcam, employing depth estimation algorithms based on focal length.
  Useful for collision avoidance and robotic navigation.
**
Tech Stack:**
  Python
  OpenCV
  YOLOv8 (Ultralytics)
  NumPy

**Run individual projects:**

SpeakSight:

  bash
    python SpeakSight.py

Depth detection:

  bash
    python depth.py
    python distance.py webcam

**Usage**

  SpeakSight.py:
    Upload or provide an image input, then receive and hear the generated description.

  depth.py/distance.py:
    Start webcam feed. Detected objects and calculated depths/distances will be visualized and/or printed for immediate usage.
