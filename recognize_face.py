import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle

def recognize_face():

    # img_path="Testing Images/PS1.jpg"
    height=200
    width=200
    X=[]

    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    print("Press 's' to start capturing images...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't access the camera.")
            break

        cv2.imshow("Press 's' to start", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'): 
            break

    cv2.destroyAllWindows()

    ret, frame = cap.read()
    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w] 
        face_resized = cv2.resize(face_roi, (200, 200))
        flatten_img=face_resized.flatten()
        X.append(flatten_img)
    
    X=np.array(X)
    
    # plt.imshow(X.reshape(height,width), cmap="gray")
    # plt.show()   

    filename="./models/model_kneighbors.pkl"
    model = pickle.load(open(filename,"rb"))

    y_predict = model.predict(X)
    print(f"Person is: {y_predict}")    

# recognize_face()