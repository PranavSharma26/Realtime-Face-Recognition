import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle

def recognize_face():

    img_path="Testing Images\PS1.jpg"
    height=200
    width=200
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
    
    X=np.array(X)
    
    # plt.imshow(X.reshape(height,width), cmap="gray")
    # plt.show()   

    filename="./models/face_recognition_model.pkl"
    model = pickle.load(open(filename,"rb"))

    y_predict = model.predict(X)
    print(f"Person is: {y_predict}")    

# recognize_face()