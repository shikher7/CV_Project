import cv2
import multiprocessing
dataset_path = 'dataset/'
trainer_path = 'trainer/'


class FaceDetector:
    def __init__(self, face_id):
        self.face_id = face_id

    def _capture_face(self):
        cam = cv2.VideoCapture(0)
        cam.set(3, 640)  # set video width
        cam.set(4, 480)  # set video height
        face_detector = cv2.CascadeClassifier('config/haarcascade_frontalface_default.xml')

        count = 0
        while (True):
            ret, img = cam.read()
            cv2.imshow('image', img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                count += 1
                cv2.imwrite(f"{dataset_path}{str(self.face_id)}/User." + str(self.face_id) + '.' + str(count) + ".jpg",
                            img)
                cv2.imshow('image', img)
            k = cv2.waitKey(100) & 0xff
            if k == 27:
                break
            elif count >= 30:
                break
        cam.release()
        cv2.destroyAllWindows()

    def capture_face(self):
        p = multiprocessing.Process(target=self._capture_face)
        p.start()
        p.join()
        return p