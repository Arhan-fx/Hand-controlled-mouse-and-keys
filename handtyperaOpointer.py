import cv2
import mediapipe as mp
import pyautogui
import screeninfo
import time
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

camera = cv2.VideoCapture(0)

screen = screeninfo.get_monitors()[0]
screen_width, screen_height = screen.width, screen.height

FINGER_KEY_MAPPING = {1: "H", 2: "E", 3: "L", 4: "O"}
prev_finger_states = {"Left": [False] * 5}
typing_delay = 0.5
last_typed_time = {"Left": [0] * 5}

calibrated_thumb_index_dist = None
calibrated_thumb_pinky_dist = None

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

def calibrate_gestures(finger_coords):
    global calibrated_thumb_index_dist, calibrated_thumb_pinky_dist
    calibrated_thumb_index_dist = calculate_distance(finger_coords[0], finger_coords[1])
    calibrated_thumb_pinky_dist = calculate_distance(finger_coords[0], finger_coords[4])

def move_mouse(x, y):
    mouse_x = int(x * screen_width)
    mouse_y = int(y * screen_height)
    pyautogui.moveTo(mouse_x, mouse_y)

def detect_gesture_right_hand(finger_coords):
    global calibrated_thumb_index_dist, calibrated_thumb_pinky_dist
    
    if calibrated_thumb_index_dist is None or calibrated_thumb_pinky_dist is None:
        calibrate_gestures(finger_coords)
        return

    thumb_index_dist = calculate_distance(finger_coords[0], finger_coords[1])
    thumb_pinky_dist = calculate_distance(finger_coords[0], finger_coords[4])

    thumb_index_close = calibrated_thumb_index_dist * 0.5
    thumb_pinky_open = calibrated_thumb_pinky_dist * 1.5

    if thumb_index_dist < thumb_index_close:
        pyautogui.click()

def detect_gesture_left_hand(finger_coords, hand_label):
    global prev_finger_states, last_typed_time

    if hand_label != "Left":
        return

    current_states = []
    for i in range(5):
        current_states.append(finger_coords[i][1] > finger_coords[0][1])

    for i, state in enumerate(current_states):
        if i in FINGER_KEY_MAPPING and state and not prev_finger_states[hand_label][i]:
            current_time = time.time()
            if current_time - last_typed_time[hand_label][i] > typing_delay:
                pyautogui.typewrite(FINGER_KEY_MAPPING[i])
                last_typed_time[hand_label][i] = current_time

    prev_finger_states[hand_label] = current_states

def gui_worker(image, hand_landmarks, hand_label):
    finger_coords = [
        (int(hand_landmarks.landmark[i].x * image.shape[1]), 
         int(hand_landmarks.landmark[i].y * image.shape[0]))
        for i in [4, 8, 12, 16, 20]
    ]
    hand_center = (
        hand_landmarks.landmark[0].x, 
        hand_landmarks.landmark[0].y
    )

    if hand_label == "Right":
        move_mouse(hand_center[0], hand_center[1])
        detect_gesture_right_hand(finger_coords)
    elif hand_label == "Left":
        detect_gesture_left_hand(finger_coords, hand_label)

prev_time = 0  

while True:
    ret, image = camera.read()
    if not ret:
        print("Failed to grab frame")
        break

    image = cv2.flip(image, 1)   
    image = cv2.resize(image, (640, 480))   
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = hand_handedness.classification[0].label
            mp_drawing.draw_landmarks(
                image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS, 
                mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2)
            )

            gui_worker(image, hand_landmarks, hand_label)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("Hand Tracking for Mouse and Keyboard Control", image)

    if cv2.waitKey(1) & 0xFF == 27:  
        break

camera.release()
cv2.destroyAllWindows()
