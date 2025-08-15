import cv2
import numpy as np
from digit_extraction import DigitExtraction

def load_data(image_path = "train_image.png", num_samples_digit = 20):
    train_img = cv2.imread(image_path)
    d = DigitExtraction(train_img, pad_value = 0)
    digits = d.extract_digits()
    y = np.array([[i for _ in range(num_samples_digit)] for i in range(10)])
    y = y.flatten()
    y = np.flip(y)
    return digits, y
