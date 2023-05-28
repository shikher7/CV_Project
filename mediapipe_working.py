import cv2
import mediapipe as mp
import numpy as np
from pynput.mouse import Controller
from screeninfo import get_monitors


class FaceMeshApplication:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mouse = Controller()
        self.monitor = get_monitors()[0]
        self.scale_factor = 10
        self.box_clicked = False
        self.calibrated = False
        self.calibration_point = (0, 0)

    def initialize(self):
        self.mouse.position = (self.monitor.width / 2, self.monitor.height / 2)
        self.calibrated = True

        cv2.namedWindow("Frame")
        cv2.setMouseCallback("Frame", self.draw_rectangle)

        self.cap = cv2.VideoCapture(0)

    def draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP and 100 < x < 200 and 100 < y < 200:
            self.box_clicked = True

    def run(self):
        with self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret: break

                frame = self.process_frame(frame, face_mesh)
                self.display_frame(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'): break

        self.cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame, face_mesh):
        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.get_face_mesh_results(rgb_image, face_mesh)
        annotated_image = self.annotate_image(rgb_image, results, width, height)

        cv2.rectangle(annotated_image, (100, 100), (200, 200), (255, 0, 0), 2)
        if self.box_clicked:
            cv2.putText(annotated_image, "Clicked!", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        return annotated_image

    def get_face_mesh_results(self, rgb_image, face_mesh):
        rgb_image.flags.writeable = False
        results = face_mesh.process(rgb_image)
        rgb_image.flags.writeable = True

        return results

    def annotate_image(self, rgb_image, results, width, height):
        annotated_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(annotated_image, face_landmarks, self.mp_face_mesh.FACEMESH_TESSELATION)
                nose_tip = face_landmarks.landmark[4]

                if self.calibrated:
                    current_point = (int(nose_tip.x * width), int(nose_tip.y * height))
                    move_x = (current_point[0] - self.calibration_point[0]) * self.scale_factor
                    move_y = (current_point[1] - self.calibration_point[1]) * self.scale_factor
                    self.mouse.move(move_x, move_y)
                    self.calibration_point = current_point

        return annotated_image

    def display_frame(self, frame):
        cv2.imshow('Frame', frame)


if __name__ == "__main__":
    app = FaceMeshApplication()
    app.initialize()
    app.run()
