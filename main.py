import cv2
import sys
import config
from hand_tracking import HandDetector   


def main():
   
    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    if not cap.isOpened():
        print(f"[ERROR] The camera is not available (index {config.CAMERA_INDEX})")
        # print("        Prueba cambiando CAMERA_INDEX en config.py")
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
    print("[INFO] MediaPipeinizialate correctly")
    print("[INFO] Push 'q' for exit\n")

    # Make that each iteration of the loop processes a new frame from the camera
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
                cv2.circle(frame, (x, y), 12, config.COLOR_YELLOW, cv2.FILLED) 

                # Texto informativo en pantalla
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

        # Status indicator 
        status_text  = "Hand detected!" if hand_detected else "Buscando mano..."
        status_color = config.COLOR_GREEN if hand_detected else config.COLOR_RED
        cv2.putText(frame, status_text, (20, real_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)

        # Screen instructions
        cv2.putText(frame, "Presiona 'q' para salir", (20, real_h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_WHITE, 1, cv2.LINE_AA)

        # Show the frame with hand landmarks and status
        cv2.imshow(config.WINDOW_NAME, frame)

        # Check for 'q' key press to exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] Closing...")
            break

    # Cleanup: release camera and close windows
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Recurses cleaned up, goodbye!")

if __name__ == "__main__":
    main()