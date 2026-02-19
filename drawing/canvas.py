# Define the disney-style magic canvas
import cv2
import numpy as np


class MagicCanvas:

    # Line colors and thicknesses for the Disney-style 
    STROKE_OUTER  = (255, 255, 255) 
    STROKE_INNER  = (255, 210, 30)   
    CURSOR_ACTIVE = (255, 230, 50)   
    CURSOR_IDLE   = (150, 150, 150)  
    OUTER_THICKNESS = 8  
    INNER_THICKNESS = 3   

    def __init__(self, width: int, height: int):
        self.width  = width
        self.height = height
        self.prev_point = None  
        self.is_drawing = False 
        self.clear()

    # Activate or deactivate drawing mode based on the gesture
    def start_stroke(self):
        self.is_drawing = True

    # Desactivate drawing mode (e.g., when the gesture is no longer active)
    def end_stroke(self):
        
        self.is_drawing = False
        self.prev_point = None   # ← "levanta el lápiz"

    # Define the method for drawing a point on the canvas
    def draw_point(self, x: int, y: int) -> None:
       
        current = (x, y)

        if self.prev_point is not None and self.is_drawing:

            cv2.line(self.canvas, self.prev_point, current,
                     self.STROKE_OUTER, self.OUTER_THICKNESS, cv2.LINE_AA)

            # Sparkerl effect: line with inner bright color
            cv2.line(self.canvas, self.prev_point, current,
                     self.STROKE_INNER, self.INNER_THICKNESS, cv2.LINE_AA)

        self.prev_point = current

    # Draw cursor
    def draw_wand_cursor(self, frame: np.ndarray, x: int, y: int, active: bool) -> None:
       
        if active:
            cv2.circle(frame, (x, y), 20, (255, 255, 255), 2,            cv2.LINE_AA)  
            cv2.circle(frame, (x, y), 10, self.CURSOR_ACTIVE, cv2.FILLED, cv2.LINE_AA) 
            cv2.circle(frame, (x, y), 4,  (255, 255, 255),  cv2.FILLED, cv2.LINE_AA) 
        else:
            cv2.circle(frame, (x, y), 8, self.CURSOR_IDLE, 2, cv2.LINE_AA)

    # Convine canvas and frame
    def overlay(self, frame: np.ndarray) -> np.ndarray:
        return cv2.add(frame, self.canvas)

    # Clean canvas and reset state of the drawing
    def clear(self) -> None:
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.prev_point = None
        self.is_drawing = False