import cv2
import faceRecognition as fr

people = ["Aakash", "Deepshikha", "Manish", "Niks", "Sagnik", "Shreya", "Ut", "Vatsi"]

if __name__ == "__main__":

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('data/model/classifier.yml')

    cap = cv2.VideoCapture(0)

    while True:

        ret, test_img = cap.read()
        faces_detected, gray_img = fr.faceDetection(test_img)

        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow("Face Detection", resized_img)
        cv2.waitKey(10)

        for face in faces_detected:

            (x, y, w, h) = face
            roi_gray = gray_img[y : y + h, x : x + w]
            label, conf = face_recognizer.predict(roi_gray)
            print(label, conf)

            fr.draw_rect(test_img, face)
            fr.put_text(test_img, people[label], x, y)

        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow("Face Detection", resized_img)
        if cv2.waitKey(10) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()