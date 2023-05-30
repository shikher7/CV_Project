import cv2
import mediapipe as mp
from pynput.mouse import Controller, Button
from screeninfo import get_monitors
from face_trainer import train


class GestureControllerApplication:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.face_detector = cv2.CascadeClassifier('config/haarcascade_frontalface_default.xml')
        self.dataset = 'dataset/'
        self.trainer = 'trainer/'
        self.mouse = Controller()
        self.monitor = get_monitors()[0]
        self.scale_factor = 10
        self.box_clicked = False
        self.calibrated = False
        self.calibration_point = (0, 0)
        self.prev_wrist_y = None

    def initialize(self):
        self.mouse.position = (self.monitor.width / 2, self.monitor.height / 2)
        self.calibrated = True
        cv2.namedWindow("Frame")
        cv2.setMouseCallback("Frame", self.draw_rectangle)
        self.cap = cv2.VideoCapture(0)

        # set video height and width
        # self.cap.set(3, 640)  # set video width
        # self.cap.set(4, 480)  # set video height


    def face_dataset(self):
        face_id = input('\n enter user id end press <return> ==>  ')
        print("\n [INFO] Initializing face capture. Look the camera and wait ...")
        count = 0

        while count < 30:

            ret, frame = self.cap.read()
            # frame = cv2.flip(frame, -1)  # flip video image vertically
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                count += 1

                # Save the captured image into the datasets folder
                cv2.imwrite(self.dataset+"User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

            k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
            if k == 27:
                break
            elif count >= 30:  # Take 30 face sample and stop video
                break


    def face_recognize(self, frame):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(self.trainer+'trainer.yml')
        face_cascade = self.face_detector
        font = cv2.FONT_HERSHEY_SIMPLEX

        id = 0
        confidence = None
        names = ['None', 'Aparna', 'Paula', 'Ilza', 'Z', 'W']
        min_w = 0.1 * self.cap.get(3)
        min_h = 0.1 * self.cap.get(4)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(min_w), int(min_h)),
        )

        for (x, y, w, h) in faces:

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            # Check if confidence is less them 100 ==> "0" is perfect match
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = None
                confidence = "  {0}%".format(round(100 - confidence))

        return id, confidence


    def draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP and 100 < x < 200 and 100 < y < 200:
            self.box_clicked = True
    def run(self):
        recognized = False
        name = confidence = None
        idx = 1
        with self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
             self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while self.cap.isOpened():

                ret, frame = self.cap.read()
                if not ret:
                    break

                if idx < 10 and recognized is False:
                    # call recognizer with frame
                    name, confidence = self.face_recognize(frame)
                    if name is not None and confidence is not None:
                        print(f"name {name} confidence{confidence}")
                        recognized = True
                elif idx >= 10 and recognized is False:
                    print('Could not recognize')
                    exit(1)
                    #display
                else:
                    frame = self.process_frame(frame, face_mesh, hands)

                cv2.putText(frame, f"User: {name}  confidence: {confidence}", (700, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                self.display_frame(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                idx += 1

        self.cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame, face_mesh, hands):
        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_mesh_results = self.get_face_mesh_results(rgb_image, face_mesh)
        hand_landmarks = self.get_hand_landmarks(rgb_image, hands)
        annotated_image = self.annotate_image(rgb_image, face_mesh_results, hand_landmarks, width, height)

        # cv2.rectangle(annotated_image, (100, 100), (200, 200), (255, 0, 0), 2)
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
                    move_x = (current_point[0] - self.calibration_point[0]) * self.scale_factor
                    move_y = (current_point[1] - self.calibration_point[1]) * self.scale_factor
                    self.mouse.move(move_x, move_y)
                    self.calibration_point = current_point

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


if __name__ == "__main__":
    app = GestureControllerApplication()
    app.initialize()

    # app.face_dataset()
    # train('dataset', 'trainer/')

    app.run()
