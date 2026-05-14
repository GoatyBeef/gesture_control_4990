Vision Mouse: Hand Gesture Control System
  This application utilizes the MediaPipe hand landmarker model to extract hand landmarks, which are fed to a 
  trained Multilayer Perceptron classifier to recognize various gestures which map to mouse inputs.  

This code contains:
  - Dual recognition modes, supporting rule-based geometric baseline and trained MLP model (gesture_recognizer.py)
  - Complete ML pipeline, including tools for data collection (data_collector.py), training (train.py), and real-time inference 
  - Smoothing, using a buffer system to prevent cursor jitter (smoother.py)
  - Is configurable, allowing for adjustments to screen resolution, click thresholds, and deadzones (config.json)

Structure:
  - main.py: The entry point of the application.
  - hand_tracker.py: Handles hand landmark detection using MediaPipe.
  - gesture_recognizer.py: Logic for identifying specific gestures (Move, Click, Scroll).
  - action_controller.py: Maps recognized gestures to system-level mouse events.
  - data_collector.py: A utility to record your own hand landmark data to a CSV.
  - train.py: Trains the PyTorch neural network on collected data.

Installation:
  Download all files in the github repo. In the terminal of your IDE, ensure all dependencies are installed.
  Our group used "pip install opencv-python mediapipe torch pyautogui numpy pandas scikit-learn" to install dependencies.

How to use:
  Path A: Using the Pre-trained Model
  Ensure gesture_model.pth and label_map.json are in the directory.
  Set USE_MODEL = True in gesture_recognizer.py.
  Run python main.py.

Path B: Training from Scratch
  Collect Data: Run python data_collector.py and follow the on-screen prompts (1-5) to record different hand poses.
  Train: Run python train.py to generate a new model and view performance metrics.
  Run: Execute python main.py.

Currently supported gestures:
  MOVE: Index finger extended.
  CLICK: Pinch gesture (index and thumb).
  SCROLL UP/DOWN: Specific multi-finger configurations.
  NO GESTURE: Simply make a fist

*I used the application "CAMO" to use my IOS device as a webcam to train and use this application, however the application works just fine using a regular webcam. The model may need to be trained and data gathered, using data_collector.py THEN train.py prior to using via main.py.
