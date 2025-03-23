import cv2
import os

# Define dataset path
dataset_path = "RealtimeDataset"

# Get user name input
user_name = input("Enter Name: ")

# Create user folder
user_path = os.path.join(dataset_path, user_name)
os.makedirs(user_path, exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("Press 's' to start capturing images...")

# Wait for user to start image capture
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

# Image capture parameters
image_count = 0
max_count = 10

print("Capturing images... Press 'q' to stop early.")

# Start capturing images
while image_count < max_count:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]  # Extract face region
        face_resized = cv2.resize(face_roi, (200, 200))  # Resize to 200x200

        # Save the detected face
        image_path = os.path.join(user_path, f"{image_count}.jpg")
        cv2.imwrite(image_path, face_resized)
        image_count += 1

    # Display the frame
    cv2.namedWindow("Capturing Frames...", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Capturing Frames...", cv2.WND_PROP_TOPMOST, 1)
    cv2.imshow("Capturing Frames...", frame)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Captured {image_count} images for {user_name}")
