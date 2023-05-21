import cv2
import mediapipe as mp
import numpy as np
from pynput.mouse import Controller, Button
from screeninfo import get_monitors
from cvzone.FaceMeshModule import FaceMeshDetector

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
detector = FaceMeshDetector(maxFaces=1)

SCALE_FACTOR = 10
mouse = Controller()
calibrated = False
calibration_point = (0, 0)
box_clicked = False
blinkCounter = 0
counter = 0
color = (255, 0, 255)
idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
ratioList = []

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

        # Blink Detection
        img, faces = detector.findFaceMesh(annotated_image, draw=False)
        if faces:
            face = faces[0]
            for id in idList:
                cv2.circle(annotated_image, face[id], 5, color, cv2.FILLED)

            leftUp = face[159]
            leftDown = face[23]
            leftLeft = face[130]
            leftRight = face[243]
            lenghtVer, _ = detector.findDistance(leftUp, leftDown)
            lenghtHor, _ = detector.findDistance(leftLeft, leftRight)

            cv2.line(annotated_image, leftUp, leftDown, (0, 200, 0), 3)
            cv2.line(annotated_image, leftLeft, leftRight, (0, 200, 0), 3)

            ratio = int((lenghtVer / lenghtHor) * 100)
            ratioList.append(ratio)
            if len(ratioList) > 3:
                ratioList.pop(0)
            ratioAvg = sum(ratioList) / len(ratioList)

            if ratioAvg < 35 and counter == 0:
                blinkCounter += 1
                color = (0, 200, 0)
                counter = 1
                # Added mouse click
                mouse.click(Button.left, 1)
            if counter != 0:
                counter += 1
                if counter > 10:
                    counter = 0
                    color = (255, 0, 255)

            cv2.putText(annotated_image, f'Blink Count: {blinkCounter}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(annotated_image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
                nose_tip = face_landmarks.landmark[4]

                if calibrated:
                    current_point = (int(nose_tip.x * width), int(nose_tip.y * height))
                    move_x = (current_point[0] - calibration_point[0]) * SCALE_FACTOR
                    move_y = (current_point[1] - calibration_point[1]) * SCALE_FACTOR
                    mouse.move(move_x, move_y)
                    calibration_point = current_point

        cv2.rectangle(annotated_image, (100, 100), (200, 200), (255, 0, 0), 2)
        if box_clicked:
            cv2.putText(annotated_image, "Clicked!", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow('Frame', annotated_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
