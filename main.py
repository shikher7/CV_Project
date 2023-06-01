import cv2
import mediapipe as mp
from pynput.mouse import Controller, Button
from screeninfo import get_monitors
import time
from threading import Thread
from SpeecToText import SpeechToText
VIDEO_WIDTH = 160
VIDEO_HEIGHT = 120



class GestureControllerApplication:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mouse = Controller()
        self.monitor = get_monitors()[0]
        self.scale_factor = 7
        self.box_clicked = False
        self.calibrated = False
        self.calibration_point = (0, 0)
        self.prev_wrist_y = None
        self.smooth_factor = 0.9  # for smoothing the mouse movement
        self.click_threshold = 0.15  # increase this threshold to reduce the sensitivity of click action
        self.last_mouse_pos = None


    def initialize(self):
        self.mouse.position = (self.monitor.width / 2, self.monitor.height / 2)
        self.calibrated = True
        cv2.namedWindow("Frame")
        cv2.setMouseCallback("Frame", self.draw_rectangle)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, VIDEO_WIDTH)  # set video width
        self.cap.set(4, VIDEO_HEIGHT)
        # self.speech_to_text = SpeechToText()

    def draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP and 100 < x < 200 and 100 < y < 200:
            self.box_clicked = True
    def run(self):
        # thread = Thread(target=self.use_speech_to_text, args=(self.speech_to_text,))
        # thread.start()
        with self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
             self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret: break

                frame = self.process_frame(frame, face_mesh, hands)
                self.display_frame(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'): break
        # thread.join()
        self.cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame, face_mesh, hands):
        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_mesh_results = self.get_face_mesh_results(rgb_image, face_mesh)
        hand_landmarks = self.get_hand_landmarks(rgb_image, hands)
        annotated_image = self.annotate_image(rgb_image, face_mesh_results, hand_landmarks, width, height)

        cv2.rectangle(annotated_image, (100, 100), (200, 200), (255, 0, 0), 2)
        if self.box_clicked:
            cv2.putText(annotated_image, "Clicked!", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        return annotated_image

    def get_face_mesh_results(self, rgb_image, face_mesh):
        rgb_image.flags.writeable = False
        results = face_mesh.process(rgb_image)
        rgb_image.flags.writeable = True
        return results

    def get_hand_landmarks(self, rgb_image, hands):
        rgb_image.flags.writeable = False
        results = hands.process(rgb_image)
        rgb_image.flags.writeable = True
        return results

    def annotate_image(self, rgb_image, face_mesh_results, hand_landmarks, width, height):
        annotated_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        if face_mesh_results.multi_face_landmarks and hand_landmarks.multi_hand_landmarks:  # check for hand landmarks here
            for face_landmarks in face_mesh_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(annotated_image, face_landmarks, self.mp_face_mesh.FACEMESH_TESSELATION)
                nose_tip = face_landmarks.landmark[4]

                if self.calibrated:
                    current_point = (int(nose_tip.x * width), int(nose_tip.y * height))
                    if self.last_mouse_pos is None:
                        self.last_mouse_pos = current_point
                    else:
                        current_point = (
                            self.smooth_factor * self.last_mouse_pos[0] + (1 - self.smooth_factor) * current_point[0],
                            self.smooth_factor * self.last_mouse_pos[1] + (1 - self.smooth_factor) * current_point[1]
                        )
                    move_x = (current_point[0] - self.calibration_point[0]) * self.scale_factor
                    move_y = (current_point[1] - self.calibration_point[1]) * self.scale_factor
                    self.mouse.move(move_x, move_y)
                    self.calibration_point = current_point
                    self.last_mouse_pos = current_point

        if hand_landmarks.multi_hand_landmarks:
            for hand_landmark in hand_landmarks.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(annotated_image, hand_landmark, self.mp_hands.HAND_CONNECTIONS)
                self.perform_mouse_click_and_scroll(hand_landmark)

        return annotated_image

    def perform_mouse_click_and_scroll(self, hand_landmark):
        thumb_tip = hand_landmark.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_finger_mcp = hand_landmark.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        index_finger_pip = hand_landmark.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        index_finger_dip = hand_landmark.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_DIP]
        index_finger_tip = hand_landmark.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        wrist = hand_landmark.landmark[self.mp_hands.HandLandmark.WRIST]

        # mouse click action
        distance = ((thumb_tip.x - index_finger_tip.x) ** 2 +
                    (thumb_tip.y - index_finger_tip.y) ** 2 +
                    (thumb_tip.z - index_finger_tip.z) ** 2) ** 0.5
        if distance < 0.1:  # you can adjust this threshold as needed
            self.mouse.click(Button.left, 1)

        # scrolling action
        # Check if index finger is pointing straight
        if index_finger_mcp.y > index_finger_pip.y > index_finger_dip.y > index_finger_tip.y:
            # Index finger is pointing upwards
            self.scroll_up()
        elif index_finger_mcp.y < index_finger_pip.y < index_finger_dip.y < index_finger_tip.y:
            # Index finger is pointing downwards
            self.scroll_down()

    def scroll_up(self):
        self.mouse.scroll(0, 1)

    def scroll_down(self):
        self.mouse.scroll(0, -1)

    def display_frame(self, frame):
        cv2.imshow('Frame', frame)

    def use_speech_to_text(self, speech_to_text):
        speech_to_text.start_recording()

        time.sleep(10000)
        speech_to_text.stop_recording()

        text = speech_to_text.get_text()

        print(text)

        speech_to_text.reset()


if __name__ == "__main__":
    app = GestureControllerApplication()
    app.initialize()
    app.run()

 # Assuming your class is in a file named "speech_to_text.py"

