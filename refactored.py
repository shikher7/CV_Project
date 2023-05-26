import cv2
import mediapipe as mp
from pynput.mouse import Controller, Button
from screeninfo import get_monitors
from cvzone.FaceMeshModule import FaceMeshDetector

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
detector = FaceMeshDetector(maxFaces=1)

SCALE_FACTOR = 10
mouse = Controller()
monitor = get_monitors()[0]
mouse.position = (monitor.width / 2, monitor.height / 2)  # center the mouse
color, rightColor = (255, 0, 255), (255, 0, 255)
blinkCounter, rightBlinkCounter = 0, 0
counter, rightCounter = 0, 0
ratioList, rightRatioList = [], []
box_clicked = False
idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
rightIdList = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]

def draw_rectangle(event, x, y, flags, param):
    global box_clicked
    if event == cv2.EVENT_LBUTTONUP and 100 < x < 200 and 100 < y < 200:
        box_clicked = True

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", draw_rectangle)
cap = cv2.VideoCapture(0)

def compute_ratio(id1, id2, id3, id4, frame, ratioList):
    up = frame[id1]
    down = frame[id2]
    left = frame[id3]
    right = frame[id4]
    lenghtVer, _ = detector.findDistance(up, down)
    lenghtHor, _ = detector.findDistance(left, right)
    ratio = int((lenghtVer / lenghtHor) * 100)
    ratioList.append(ratio)
    if len(ratioList) > 3: ratioList.pop(0)
    return sum(ratioList) / len(ratioList)

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)
        annotated_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        img, faces = detector.findFaceMesh(annotated_image, draw=False)

        if faces:
            face = faces[0]
            ratioAvg = compute_ratio(159, 23, 130, 243, face, ratioList)
            rightRatioAvg = compute_ratio(386, 253, 463, 359, face, rightRatioList)

            for id in idList + rightIdList: cv2.circle(annotated_image, face[id], 5, color if id in idList else rightColor, cv2.FILLED)

            if ratioAvg < 35 and not counter:
                blinkCounter += 1
                color = (0, 200, 0)
                counter = 1
                mouse.click(Button.left, 1)

            if rightRatioAvg < 35 and not rightCounter:
                rightBlinkCounter += 1
                rightColor = (0, 200, 0)
                rightCounter = 1
                mouse.click(Button.left, 1)

            if counter: counter = 0 if counter > 10 else counter + 1; color = (255, 0, 255) if not counter else color
            if rightCounter: rightCounter = 0 if rightCounter > 10 else rightCounter + 1; rightColor = (255, 0, 255) if not rightCounter else rightColor

            cv2.putText(annotated_image, f'Blink Count: {blinkCounter}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(annotated_image, f'Right Blink Count: {rightBlinkCounter}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, rightColor, 2)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(annotated_image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
                nose_tip = face_landmarks.landmark[4]
                current_point = (int(nose_tip.x * width), int(nose_tip.y * height))
                mouse.move(*[(current_point[i] - mouse.position[i]) * SCALE_FACTOR for i in range(2)])
                mouse.position = current_point

        cv2.rectangle(annotated_image, (100, 100), (200, 200), (255, 0, 0), 2)
        if box_clicked: cv2.putText(annotated_image, "Clicked!", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow('Frame', annotated_image)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
