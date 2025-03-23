import cv2
import os

def recognize_face():

    img_path="Testing Images\Profile.jpg"

    X=[]

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    image=cv2.imread(img_path)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w] 
            face_resized = cv2.resize(face_roi, (200, 200))
            flatten_img=face_resized.flatten()
            X.append(flatten_img)
    
    return X

# recognize_face()