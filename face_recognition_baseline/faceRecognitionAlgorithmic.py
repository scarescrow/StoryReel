import cv2
import os
import numpy as np

people = {
    'Aakash': 0,
    'Deepshikha': 1,
    'Manish': 2,
    'Niks': 3,
    'Sagnik': 4,
    'Shreya': 5,
    'Ut': 6,
    'Vatsi': 7
}

def faceDetection(test_img):
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    face_haar_cascade = cv2.CascadeClassifier('assets/haarcascade_frontalface_default.xml')
    faces = face_haar_cascade.detectMultiScale(gray_img, scaleFactor = 1.3, minNeighbors = 5)

    return faces, gray_img

def labels_for_training_data(directory):

    faces = []
    faceID = []

    for path, subdirs, filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("Skipping system file {}".format(filename))
                continue

            img_id = people[os.path.basename(path)]
            img_path = os.path.join(path, filename)
            print("Processing image ", img_path, "id: ", img_id)
            current_img = cv2.imread(img_path)
            if current_img is None:
                print("Unable to load image")
                continue

            faces_rect, gray_img = faceDetection(current_img)
            if len(faces_rect) != 1:
                print(len(faces_rect), "faces found in this pic!")
                continue

            (x, y, w, h) = faces_rect[0]

            roi_gray = gray_img[y : y + w, x : x + h]

            faces.append(roi_gray)
            faceID.append(img_id)


    return faces, faceID

def train_classifier(faces, faceID):

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(faceID))

    return face_recognizer

def draw_rect(test_img, face):

    (x, y, w, h) = face
    cv2.rectangle(test_img, (x, y), (x + w, y + h), (0, 255, 0), thickness = 5)

def put_text(test_img, text, x, y):
    cv2.putText(test_img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 5, (0, 255, 0), 6)
