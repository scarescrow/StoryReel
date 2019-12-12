import cv2
import os
import numpy as np
import faceRecognition as fr
import sys

people = ["Aakash", "Deepshikha", "Manish", "Niks", "Sagnik", "Shreya", "Ut", "Vatsi"]

if __name__ == "__main__":

    try:
        img = sys.argv[1]
    except:
        img = 1

    test_img = cv2.imread('data/images/test/{}.jpg'.format(img))

    faces_detected, gray_img = fr.faceDetection(test_img)

    print(len(faces_detected))

    # for (x, y, w, h) in faces_detected:
    #     cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness = 5)

    faces, face_ids = fr.labels_for_training_data("data/images/train")

    face_recognizer = fr.train_classifier(faces, face_ids)
    # face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    # face_recognizer.read("data/model/classifier.yml")

    face_recognizer.save('data/model/classifier.yml')

    for face in faces_detected:

        (x, y, w, h) = face
        roi_gray = gray_img[y : y + h, x : x + w]
        label, conf = face_recognizer.predict(roi_gray)
        print(label, conf)

        fr.draw_rect(test_img, face)
        if conf <= 40:
            fr.put_text(test_img, people[label], x, y)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow("Face Detection", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


