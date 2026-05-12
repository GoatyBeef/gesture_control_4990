import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandTracker:
    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.5):
        # Download the hand landmarker model (required for Tasks API)
        model_path = self._download_model()
        
        # Create options for the hand landmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=tracking_confidence,
            min_tracking_confidence=tracking_confidence
        )
        
        # Create the hand landmarker
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.results = None
        
    def _download_model(self):
        """Download the hand landmarker model if not present."""
        import urllib.request
        import os
        
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            print("Downloading MediaPipe hand landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("Model downloaded successfully!")
        return model_path
    
    def find_hands(self, frame, draw=True):
        """Process the frame and find hands."""
        # Convert to RGB and then to MediaPipe Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hand landmarks
        self.results = self.detector.detect(mp_image)
        
        # Draw landmarks if requested
        if draw and self.results.hand_landmarks:
            for hand_landmarks in self.results.hand_landmarks:
                # Draw connections (this is simplified - you can add full skeleton later)
                for landmark in hand_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        
        return frame
    
    def get_landmarks(self, frame):
        """Get the list of landmark positions for the first hand detected."""
        landmarks = []
        if self.results and self.results.hand_landmarks and len(self.results.hand_landmarks) > 0:
            hand = self.results.hand_landmarks[0]
            h, w = frame.shape[:2]
            
            for lm in hand:
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                landmarks.append((cx, cy))
        return landmarks