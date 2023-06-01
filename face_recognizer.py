from flask import Flask
import cv2
import multiprocessing
import time
import json

dataset_path = 'dataset/'
trainer_path = 'trainer/'

try:
    with open('users_db.json', 'r') as f:
        users_db = json.load(f)
except FileNotFoundError:
    users_db = {}  # If the file doesn't exist yet, start with an empty database


class FaceRecognizer:
    VIDEO_WIDTH = 640
    VIDEO_HEIGHT = 480
    TIMEOUT = 3
    ESC_KEY = 27

    def __init__(self, trainer_path):
        self.trainer_path = trainer_path
        self.id = None  # set default value for id
        self.confidence = 100  # set default value for confidence

    def recognize_face(self):
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        p = multiprocessing.Process(target=self._recognize_face, args=(return_dict,))
        p.start()
        p.join()
        self.id, self.confidence = return_dict['id'], return_dict['confidence']
        return self
    def _recognize_face(self, return_dict):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(self.trainer_path + 'trainer.yml')


        faceCascade = cv2.CascadeClassifier('config/haarcascade_frontalface_default.xml')

        cam = cv2.VideoCapture(0)
        cam.set(3, self.VIDEO_WIDTH)  # set video width
        cam.set(4, self.VIDEO_HEIGHT)  # set video height
        minW = 0.1 * cam.get(3)
        minH = 0.1 * cam.get(4)
        start_time = time.time()

        while time.time() - start_time < self.TIMEOUT:
            # print(f'recognizing face, elapsed time: {time.time() - start_time} secs')
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH)),
            )
            for (x, y, w, h) in faces:
                id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(img, str(users_db[str(id)]["name"]), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow('image', img)
                print(confidence, id)
                if confidence > 10 and id is not None:
                    return_dict['id'], return_dict['confidence'] = id, confidence
            if cv2.waitKey(100) & 0xff == self.ESC_KEY:
                break
        cam.release()
        cv2.destroyAllWindows()
        print(f'Face recognized with confidence ',return_dict['id'], return_dict['confidence'])
        return