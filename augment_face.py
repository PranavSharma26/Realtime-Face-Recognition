import cv2
import os
import numpy as np
import random

def add_noise(image):
    row, col = image.shape
    mean = 0
    sigma = 0.7
    gauss = np.random.normal(0, sigma, (row, col)).astype('uint8')
    noisy = cv2.add(image, gauss)
    return noisy

def adjust_brightness(image, factor=1.2):
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def rotate_image(image, angle=10):
    h, w = image.shape
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, (w, h))

def blur_image(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def augment_face(image):
    augmented = []
    augmented.append(image)  # Original

    augmented.append(cv2.flip(image, 1))  # Horizontal flip
    augmented.append(add_noise(image))  # Gaussian noise
    augmented.append(adjust_brightness(image, 1.5))  # Brighten
    augmented.append(adjust_brightness(image, 0.5))  # Darken
    augmented.append(rotate_image(image, 5))  # Rotate +5 deg
    augmented.append(rotate_image(image, -5))  # Rotate -5 deg
    augmented.append(blur_image(image))  # Slight blur
    augmented.append(cv2.equalizeHist(image))  # Histogram equalization

    return augmented