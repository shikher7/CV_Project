import pyaudio
import websockets
import asyncio
import base64
import json
import logging
import cv2
import mediapipe as mp
from pynput.mouse import Controller, Button
from screeninfo import get_monitors
import threading



class SpeechToText:
    FRAMES_PER_BUFFER = 3200
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    URL = "wss://api.assemblyai.com/v2/realtime/ws?sample_rate=16000"
    AUTH_KEY = "fa26bc637ac94d7bbf5b0139a97a77a0"  # Insert your AssemblyAI key here


    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.text = ""
        self.is_speaking = False
        self._ws = None
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def start_recording(self):
        self.is_speaking = True
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.FRAMES_PER_BUFFER
        )

        self.loop.run_until_complete(self.send_receive())

    def stop_recording(self):
        self.is_speaking = False
        self.stream.stop_stream()
        self.stream.close()

    async def send_receive(self):
        print(f'Connecting websocket to url {self.URL}')
        async with websockets.connect(self.URL, extra_headers=(("Authorization", self.AUTH_KEY),),
                                      ping_interval=5, ping_timeout=20) as _ws:
            self._ws = _ws
            await asyncio.sleep(0.1)
            print("Receiving SessionBegins ...")
            session_begins = await _ws.recv()
            print(session_begins)
            print("Sending messages ...")

            tasks = [self.loop.create_task(self.send()), self.loop.create_task(self.receive())]
            await asyncio.wait(tasks)

    async def send(self):
        while self.is_speaking:
            try:
                data = self.stream.read(self.FRAMES_PER_BUFFER)
                data = base64.b64encode(data).decode("utf-8")
                json_data = json.dumps({"audio_data": str(data)})
                await self._ws.send(json_data)
            except websockets.exceptions.ConnectionClosedError as e:
                logging.exception(f'Websocket closed {e.code}')
                break
            except Exception:
                logging.exception('Not a websocket')
            await asyncio.sleep(0.01)

    async def receive(self):
        while self.is_speaking:
            try:
                result_str = await self._ws.recv()
                self.text += json.loads(result_str)['text']
                print(self.text)
            except websockets.exceptions.ConnectionClosedError as e:
                logging.exception(f'Websocket closed {e.code}')
                break
            except Exception:
                logging.exception('Not a websocket')

    def transcribe(self):
        return self.text

class GestureControllerApplication:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mouse = Controller()
        self.monitor = get_monitors()[0]
        self.scale_factor = 10
        self.box_clicked = False
        self.calibrated = False
        self.calibration_point = (0, 0)
        self.prev_wrist_y = None
        self.stt = SpeechToText()
        self.is_speaking = False


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
        with self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
             self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while self.cap.isOpened():
                ret, frame = self.cap.read()

                if not ret: break

                frame = self.process_frame(frame, face_mesh, hands)
                self.display_frame(frame)
                # get count of frames in video 0,1,2,3....
                if cv2.waitKey(1) & 0xFF == ord('q'): break

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

                upper_lip = face_landmarks.landmark[61]
                lower_lip = face_landmarks.landmark[291]
                mouth_open = ((upper_lip.x - lower_lip.x) ** 2 + (upper_lip.y - lower_lip.y) ** 2 + (
                            upper_lip.z - lower_lip.z) ** 2) ** 0.5 > 0.03
                if mouth_open and not self.is_speaking:
                    print('Start speaking')
                    self.is_speaking = True
                    threading.Thread(target=self.stt.start_recording).start()

                if not mouth_open and self.is_speaking:
                    print('Stop speaking')
                    self.is_speaking = False
                    threading.Thread(target=self.stt.stop_recording).start()
                    print(f"Transcription: {self.stt.transcribe()}")

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
    app.run()
