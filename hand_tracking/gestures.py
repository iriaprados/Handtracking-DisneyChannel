#  File for gesture recognition logic based on hand landmarks from MediaPipe

# Logic for recognizing which fingers are up or down based on landmark positions
FINGER_LANDMARKS = {
    "index":  (8,  6),
    "middle": (12, 10),
    "ring":   (16, 14),
    "pinky":  (20, 18),
}

THUMB_TIP = 4
THUMB_IP  = 3

# Function for determining which fingers are up based on landmark positions
def get_fingers_state(landmarks: list) -> dict:
    
    if len(landmarks) < 21: # If there are not enough landmarks, return all fingers as down
        return {name: False for name in ["index", "middle", "ring", "pinky", "thumb"]}

    state = {}

    for name, (tip_id, pip_id) in FINGER_LANDMARKS.items():
        tip_y = landmarks[tip_id][1]   # Y de la punta
        pip_y = landmarks[pip_id][1]   # Y del nudillo medio
        state[name] = tip_y < pip_y    # True = levantado

    # El pulgar se mueve en X, no en Y
    thumb_tip_x = landmarks[THUMB_TIP][0]
    thumb_ip_x  = landmarks[THUMB_IP][0]
    state["thumb"] = abs(thumb_tip_x - thumb_ip_x) > 20

    return state

# Function for defineing the "magic wand" gesture based on fingers state
def is_wand_gesture(landmarks: list) -> bool:
    
    state = get_fingers_state(landmarks)

    return (
        state["index"]
        and not state["middle"]
        and not state["ring"]
        and not state["pinky"]
    )