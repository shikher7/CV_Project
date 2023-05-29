import cv2
import numpy as np
from PIL import Image
import os

class FaceDetector:
    def __init__(self, face_id):
        self.face_id = face_id
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, 640) # set video width
        self.cam.set(4, 480) # set video height
        self.face_detector = cv2.CascadeClassifier('config/haarcascade_frontalface_default.xml')

    def capture_face(self):
        count = 0
        while(True):
            ret, img = self.cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
                count += 1
                cv2.imwrite("dataset/User." + str(self.face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
                cv2.imshow('image', img)
            k = cv2.waitKey(100) & 0xff
            if k == 27:
                break
            elif count >= 30:
                 break
        self.cam.release()
        cv2.destroyAllWindows()

class FaceTrainer:
    def __init__(self, path, output):
        self.path = path
        self.output = output

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
        print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        faces, ids = self.getImagesAndLabels(self.path)
        recognizer.train(faces, np.array(ids))
        recognizer.write(self.output+'trainer.yml')
        print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

if __name__ == "__main__":
    # face_id = input('\n enter user id end press <return> ==>  ')
    # fd = FaceDetector(face_id)
    # fd.capture_face()

    path_to_images = 'dataset'
    output_directory = 'trainer/'
    ft = FaceTrainer(path_to_images, output_directory)
    ft.train()
