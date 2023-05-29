import cv2
import numpy as np
from PIL import Image
import os

dataset_path = 'dataset/'
trainer_path = 'trainer/'
class FaceTrainer:
    def __init__(self, path, output):
        self.path = path
        self.output = output
        self.id = None
        self.confidence = None

    @staticmethod
    def getImagesAndLabels(path):
        detector = cv2.CascadeClassifier("config/haarcascade_frontalface_default.xml")
        faceSamples = []
        ids = []

        for dir_name in os.listdir(path):  # iterate through each folder named as '0', '1', '2', etc.
            dir_path = os.path.join(path, dir_name)
            if os.path.isdir(dir_path):
                imagePaths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.jpg')]
                for imagePath in imagePaths:
                    PIL_img = Image.open(imagePath).convert('L')
                    img_numpy = np.array(PIL_img, 'uint8')
                    id = os.path.split(imagePath)[-1].split(".")[1]
                    faces = detector.detectMultiScale(img_numpy)
                    for (x, y, w, h) in faces:
                        faceSamples.append(img_numpy[y:y + h, x:x + w])
                        ids.append(int(dir_name))  # here we're storing the id as the folder name
        return faceSamples, ids

    def train(self):
        print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
        recognizer1 = cv2.face.LBPHFaceRecognizer_create()
        faces, ids = self.getImagesAndLabels(self.path)
        recognizer1.train(faces, np.array(ids))
        recognizer1.write(self.output + 'trainer.yml')
        print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))


