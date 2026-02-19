import cv2
import sys
import config
from hand_tracking import HandDetector
from hand_tracking.gestures import is_wand_gesture
from drawing import MagicCanvas


def draw_ui(frame, real_w, real_h, hand_detected, wand_active, show_debug):
    
    # Wand status l
    if wand_active:
        label = " WAND ACTIVE"
        color = (50, 220, 255)   
    elif hand_detected:
        label = " hAND DETECTED — UPLIFT INDEX TO DRAW "
        color = config.COLOR_GREEN
    else:
        label = " FIND YOUR HAND AND LIFT YOUR INDEX FINGER TO DRAW "
        color = config.COLOR_RED

    cv2.putText(frame, label, (20, real_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    # Instructions
    instructions = "'q': exit | 'c': clean | 'd': debug"
    cv2.putText(frame, instructions, (20, real_h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, config.COLOR_WHITE, 1, cv2.LINE_AA)

    # Debug mode status
    if show_debug:
        cv2.putText(frame, "DEBUG ON", (real_w - 130, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_GREEN, 2, cv2.LINE_AA)


def main():
    
    # Inizialize camera
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] It is not possible to open the camera (index {config.CAMERA_INDEX})")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          config.FPS_TARGET)

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera open— {real_w}x{real_h}")

    # Inizialize modules
    detector   = HandDetector()
    canvas     = MagicCanvas(real_w, real_h)
    show_debug = False   # toggle con tecla 'd'

    print("[INFO] Done. UPLIFT INDEX TO DRAW.")
    print("[INFO] 'q' exit | 'c' clean | 'd' debug landmarks\n")

    # Main loop
    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)   # espejo horizontal

        # Hand detection and landmarks extraction
        hand_detected = detector.find_hands(frame, draw=show_debug)

        wand_active = False

        if hand_detected:
            landmarks = detector.get_all_landmarks(frame) # 21 landmarks in pixel coordinates [(x, y), ...]
            wand_active = is_wand_gesture(landmarks) 
            tip = detector.get_landmark_pos(frame, config.INDEX_TIP)

            if tip: # If the index tip is detected, we can draw
                x, y = tip

                if wand_active:
                    # Activate drawing mode and draw on the canvas
                    canvas.start_stroke()
                    canvas.draw_point(x, y)
                else:
                    # Unvalid gesture → lift the pen to avoid connecting strokes
                    canvas.end_stroke()

                canvas.draw_wand_cursor(frame, x, y, wand_active)

        else:
            canvas.end_stroke()

        # Fusion canvas and frame
        frame = canvas.overlay(frame)

      
        draw_ui(frame, real_w, real_h, hand_detected, wand_active, show_debug)
        cv2.imshow(config.WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            canvas.clear()
            print("[INFO] Canvas limpiado")
        elif key == ord("d"):
            show_debug = not show_debug
            print(f"[INFO] Debug landmarks: {'ON' if show_debug else 'OFF'}")

    # Clean information
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Close.")


if __name__ == "__main__":
    main()