import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_model():
    path="RealtimeDataset"

    X=[]
    y=[]
    person_names=[]

    height=200
    width=200

    for person in os.listdir(path):
        person_path=os.path.join(path,person)

        if os.path.isdir(person_path):
            person_names.append(person)

            for img_name in os.listdir(person_path):
                img_path=os.path.join(person_path,img_name)
                img=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
                
                if(img is not None):
                    img=cv2.resize(img,(200,200))
                    img_flatten = img.flatten()
                    X.append(img_flatten)
                    y.append(person)
                    
    X=np.array(X)
    y=np.array(y)

    # print(X.shape)
    # print(y.shape)

    # plt.imshow(X[45].reshape(height,width), cmap="gray")
    # plt.show()

    mean_face=np.mean(X,axis=0)
    X_centered = X - mean_face

    # plt.imshow(mean_face.reshape(height,width), cmap="gray")
    # plt.title("Mean Face")
    # plt.show()

    # plt.imshow(X_centered[0].reshape(height,width), cmap="gray")
    # plt.title("Mean Face")
    # plt.show()

    n_component=50
    pca=PCA(n_component)
    X_pca=pca.fit_transform(X_centered)

    eigenfaces = pca.components_

    print(f"Eigenfaces shape: {eigenfaces.shape}")
    fig, axes = plt.subplots(8, 5, figsize=(10,10))

    # for i, ax in enumerate(axes.flat):
    #     ax.imshow(eigenfaces[i].reshape(height,width), cmap='gray')
    #     ax.set_title(f"Eigenface {i+1}")
    #     ax.axis('off')
    # plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, stratify=y,random_state=42)

    knn=KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    knn.fit(X_train,y_train)

    y_pred=knn.predict(X_test)

    accuracy=accuracy_score(y_test,y_pred)
    confusion=confusion_matrix(y_test,y_pred)
    report=classification_report(y_test,y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix: \n{confusion}")
    print(f"Classification Report: \n{report}")

    # with open("face_recognition_model.pkl","wb") as f:
    #     pickle.dump(knn,f)

# train_model()