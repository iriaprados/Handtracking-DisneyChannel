import cv2
import numpy as np
import sys
import config
from hand_tracking import HandDetector   


def main():
   
    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    if not cap.isOpened():
        print(f"[ERROR] The camera is not available (index {config.CAMERA_INDEX})")
        sys.exit(1)

    # Configuration for the camera capture
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          config.FPS_TARGET)

    # Verify the actual resolution of the camera 
    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera open - Resolution: {real_w}x{real_h}")

    # Initialize the hand detector
    detector = HandDetector()
    print("[INFO] MediaPipe initialized correctly")
    print("[INFO] Press 'q' to exit, 'c' to clear canvas\n")

    # Drawing with canvas setup
    canvas = np.zeros((real_h, real_w, 3), dtype=np.uint8) 
    previous_point = None  
    
    while True:
        # Capture a new frame
        success, frame = cap.read()

        if not success: # Error on frame capture
            print("[WARN] Frame error or empty frame received, skipping...")
            continue
        
        # Horizantal capture for mirror effect - as a selfie 
        frame = cv2.flip(frame, 1) # Num. 1 define horizontal capture 

        # Hand detection and drawing landmarks on the frame
        hand_detected = detector.find_hands(frame, draw=True)

        # If a hand is detected, get the position of the index finger 
        if hand_detected:
            pos = detector.get_landmark_pos(frame, config.INDEX_TIP)
            if pos:
                x, y = pos
                
                # Draw a line from the previous point to the current point
                if previous_point is not None:
                    cv2.line(canvas, previous_point, (x, y), config.COLOR_BLUE, 3)
                
                # Update the previous point to the current position
                previous_point = (x, y)
                
                # Circle at the current position of the index finger
                cv2.circle(frame, (x, y), 12, config.COLOR_YELLOW, cv2.FILLED) 
                
                # Informative text on the screen 
                cv2.putText(
                    frame,
                    f"Indice: ({x}, {y})",
                    (20, 50),              # text position (x, y)
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,                  
                    config.COLOR_YELLOW,
                    2,                    
                    cv2.LINE_AA            
                )
        # else:
        #     # Si no hay mano, "levantar el lápiz" (resetear punto anterior)
        #     previous_point = None

        # Superpose the canvas with the drawings on the original frame
        frame = cv2.add(frame, canvas)
        
        # Status indicator 
        status_text  = "Hand detected!" if hand_detected else "Buscando mano..."
        status_color = config.COLOR_GREEN if hand_detected else config.COLOR_RED
        cv2.putText(frame, status_text, (20, real_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)

        # Screen instructions
        cv2.putText(frame, "'q': salir | 'c': limpiar canvas", (20, real_h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_WHITE, 1, cv2.LINE_AA)

        # Show the frame with hand landmarks and status
        cv2.imshow(config.WINDOW_NAME, frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] Closing...")
            break
        elif key == ord("c"):
            # Clear the canvas and reset previous point
            canvas = np.zeros((real_h, real_w, 3), dtype=np.uint8)
            previous_point = None
            print("[INFO] Canvas cleared!")

    # Cleanup: release camera and close windows
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Recurses cleaned up")

if __name__ == "__main__":
    main()