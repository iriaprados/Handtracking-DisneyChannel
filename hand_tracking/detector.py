# --- Handtracking detector using MediaPipe and OpenCV ---

import cv2
from matplotlib.pyplot import draw
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import config
import os
import urllib.request

class HandDetector:

    def __init__(self):
        # Download model if not exists
        model_path = 'hand_landmarker.task'
        if not os.path.exists(model_path):
            print("[INFO] Downloading hand landmarker model...")
            model_url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
            urllib.request.urlretrieve(model_url, model_path)
            print("[INFO] Model downloaded successfully")
        
        # Hand tracking model initialization using new API
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=config.MAX_HANDS,
            min_hand_detection_confidence=config.DETECTION_CONFIDENCE,
            min_tracking_confidence=config.TRACKING_CONFIDENCE,
            running_mode=vision.RunningMode.VIDEO
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.results = None # Store results
        self.frame_timestamp_ms = 0
        
        # Hand connections for drawing
        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ] 
    
    # Enpoint for finding hands in a frame
    def find_hands(self, frame: "mp.ndarray", draw: bool = True) -> bool:
    
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        self.results = self.detector.detect_for_video(mp_image, self.frame_timestamp_ms)
        self.frame_timestamp_ms += 1

        if self.results.hand_landmarks:
            if draw:
                self._draw_landmarks(frame)
            return True

        return False
    
    # Endpoint for get the landmark position in pixel coordinates
    def get_landmark_pos(self, frame: "mp.ndarray", landmark_id: int) -> tuple[int, int] | None:
        
        # If the hand is not detected or there are no landmarks 
        if not self.results or not self.results.hand_landmarks:
            return None

        hand = self.results.hand_landmarks[0]
        lm = hand[landmark_id]

        # Convert normalized coordinates to pixel coordinates
        h, w, _ = frame.shape # height, width, channels
        x = int(lm.x * w)
        y = int(lm.y * h)

        return x, y
    
    # Enpoint for get all landmarks in a list of tuples (x, y) - coordinates in pixels
    def get_all_landmarks(self, frame: "mp.ndarray") -> list[tuple[int, int]]:
       
        if not self.results or not self.results.hand_landmarks: 
            return []

        hand = self.results.hand_landmarks[0]
        h, w, _ = frame.shape # Recover height and width

        landmarks = [] 
        for lm in hand:
            x = int(lm.x * w)
            y = int(lm.y * h)
            landmarks.append((x, y))

        return landmarks  # landmarks points 21 on total 

    # Enpoint for drawing landmarks and connections on the frame
    def _draw_landmarks(self, frame: "mp.ndarray") -> None:
       
        # Define the connections between landmarks
        h, w, _ = frame.shape
        for hand_landmarks in self.results.hand_landmarks:
            # Draw connections
            for connection in self.HAND_CONNECTIONS:
                start_idx, end_idx = connection
                start_point = (int(hand_landmarks[start_idx].x * w), int(hand_landmarks[start_idx].y * h))
                end_point = (int(hand_landmarks[end_idx].x * w), int(hand_landmarks[end_idx].y * h))
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
            
            # Draw landmarks as circles
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
