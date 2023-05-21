import cv2
import mediapipe as mp
import numpy as np
from pynput.mouse import Controller, Button
from scipy.spatial import distance as dist
from screeninfo import get_monitors


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
SCALE_FACTOR = 10
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3
EYE_OPEN_CLOSE_THRESH = 0.05
COUNTER_LEFT = 0
COUNTER_RIGHT = 0
COUNTER_BOTH = 0
TOTAL_LEFT = 0
TOTAL_RIGHT = 0
TOTAL_BOTH = 0
calibrated = False
calibration_point = (0, 0)
mouse = Controller()
box_clicked = False

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def eye_open_close_distance(eye):
    upper = (eye[1] + eye[2]) / 2
    lower = (eye[4] + eye[5]) / 2
    return dist.euclidean(upper, lower)





def draw_rectangle(event, x, y, flags, param):
    global box_clicked
    if event == cv2.EVENT_LBUTTONUP:
        if x > 100 and x < 200 and y > 100 and y < 200:
            box_clicked = True


# get screen size
monitor = get_monitors()[0]
screen_width = monitor.width
screen_height = monitor.height

# center the mouse
mouse.position = (screen_width / 2, screen_height / 2)
calibrated = True

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", draw_rectangle)

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        results = face_mesh.process(rgb_image)
        rgb_image.flags.writeable = True
        annotated_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(annotated_image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
                left_eye_landmarks = [face_landmarks.landmark[i] for i in range(130, 136)]
                right_eye_landmarks = [face_landmarks.landmark[i] for i in range(373, 379)]
                nose_tip = face_landmarks.landmark[4]
                leftEye = np.array([(int(landmark.x * width), int(landmark.y * height)) for landmark in left_eye_landmarks])
                rightEye = np.array([(int(landmark.x * width), int(landmark.y * height)) for landmark in right_eye_landmarks])
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                leftEyeDistance = eye_open_close_distance(leftEye)
                rightEyeDistance = eye_open_close_distance(rightEye)
                distance = (leftEyeDistance + rightEyeDistance) / 2.0
                if ear < EYE_AR_THRESH or distance < EYE_OPEN_CLOSE_THRESH:
                    print("Blinking")
                    COUNTER_LEFT += leftEAR < EYE_AR_THRESH or leftEyeDistance < EYE_OPEN_CLOSE_THRESH
                    COUNTER_RIGHT += rightEAR < EYE_AR_THRESH or rightEyeDistance < EYE_OPEN_CLOSE_THRESH
                    COUNTER_BOTH += ear < EYE_AR_THRESH or distance < EYE_OPEN_CLOSE_THRESH
                else:
                    if COUNTER_LEFT >= EYE_AR_CONSEC_FRAMES:
                        print("Left eye blinked!")
                        TOTAL_LEFT += 1
                        cv2.putText(annotated_image, "Left eye blinked!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        calibrated = True
                        calibration_point = (int(nose_tip.x * width), int(nose_tip.y * height))
                    if COUNTER_RIGHT >= EYE_AR_CONSEC_FRAMES:
                        print("Right eye blinked!")
                        TOTAL_RIGHT += 1
                        cv2.putText(annotated_image, "Right eye blinked!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        mouse.click(Button.left, 1)
                    if COUNTER_BOTH >= EYE_AR_CONSEC_FRAMES:
                        print("Both eyes blinked!")
                        TOTAL_BOTH += 1
                        cv2.putText(annotated_image, "Both eyes blinked!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cap.release()
                        cv2.destroyAllWindows()
                        exit(0)
                    COUNTER_LEFT = 0
                    COUNTER_RIGHT = 0
                    COUNTER_BOTH = 0
                print(calibrated)
                if calibrated:
                    current_point = (int(nose_tip.x * width), int(nose_tip.y * height))
                    move_x = (current_point[0] - calibration_point[0]) * SCALE_FACTOR
                    move_y = (current_point[1] - calibration_point[1]) * SCALE_FACTOR
                    mouse.move(move_x, move_y)
                    calibration_point = current_point

        cv2.rectangle(annotated_image, (100, 100), (200, 200), (255, 0, 0), 2)
        if box_clicked:
            print("Clicked!")
            cv2.putText(annotated_image, "Clicked!", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow('Frame', annotated_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
