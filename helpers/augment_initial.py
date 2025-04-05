import os
import sys
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from augment_face import *

dataset_path = "RealtimeDataset"

for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_path):
        continue

    print(f"Augmenting images for: {person_name}")

    for img_name in os.listdir(person_path):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg", ".pgm")):
            continue

        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Failed to read {img_path}")
            continue

        img = cv2.resize(img, (200, 200))

        augmented_images = augment_face(img)

        base_name = os.path.splitext(img_name)[0]
        for idx, aug_img in enumerate(augmented_images[1:], start=1):
            new_name = f"{base_name}_aug_{idx}.jpg"
            new_path = os.path.join(person_path, new_name)
            cv2.imwrite(new_path, aug_img)

    print(f"Augmentation done for: {person_name}")
