# --- Handtracking detector using MediaPipe and OpenCV ---

import cv2
from matplotlib.pyplot import draw
import mediapipe as mp
import config

class HandDetector:

    def __init__(self):
        # Modele for hand detection and tracking
        self.mp_hands    = mp.solutions.hands
        # Drawing tools from MediaPipe 
        self.mp_draw     = mp.solutions.drawing_utils 
        self.mp_styles   = mp.solutions.drawing_styles

        # Hand tracking model initialization 
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.MAX_HANDS, # 1 
            min_detection_confidence=config.DETECTION_CONFIDENCE,
            min_tracking_confidence=config.TRACKING_CONFIDENCE,
        )

        self.results = None # Store results 
    
    # Enpoint for finding hands in a frame
    def find_hands(self, frame: "mp.ndarray", draw: bool = True) -> bool:
    
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_frame) 

        if self.results.multi_hand_landmarks:
            if draw:
                self._draw_landmarks(frame)
            return True

        return False
    
    # Endpoint for get the landmark position in pixel coordinates
    def get_landmark_pos(self, frame: "mp.ndarray", landmark_id: int) -> tuple[int, int] | None:
        
        # If the hand is not detected or there are no landmarks 
        if not self.results or not self.results.multi_hand_landmarks:
            return None

        hand = self.results.multi_hand_landmarks[0]
        lm = hand.landmark[landmark_id]

        # Convert normalized coordinates to pixel coordinates
        h, w, _ = frame.shape # height, width, channels
        x = int(lm.x * w)
        y = int(lm.y * h)

        return x, y
    
    # Enpoint for get all landmarks in a list of tuples (x, y) - coordinates in pixels
    def get_all_landmarks(self, frame: "mp.ndarray") -> list[tuple[int, int]]:
       
        if not self.results or not self.results.multi_hand_landmarks: 
            return []

        hand = self.results.multi_hand_landmarks[0]
        h, w, _ = frame.shape # Recover height and width

        landmarks = [] 
        for lm in hand.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            landmarks.append((x, y))

        return landmarks  # landmarks points 21 on total 

    # Enpoint for drawing landmarks and connections on the frame
    def _draw_landmarks(self, frame: "mp.ndarray") -> None:
       
        # Define the connections between landmarks     
        for hand_landmarks in self.results.multi_hand_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,         
                self.mp_styles.get_default_hand_landmarks_style(), 
                self.mp_styles.get_default_hand_connections_style(), 
            )

