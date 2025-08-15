import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter

class DigitExtraction:

    def __init__(self, image, sigma = 1, image_size = 28, pad_value = 10):
        self.I = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.uint8) if image.ndim == 3 else image
        self.image = image
        self.sigma = sigma
        self.size = image_size
        self.pad_value = pad_value
        self.bounding_boxes = []

    def find_contours(self):
        blurred_img = gaussian_filter(self.I, sigma = self.sigma)
        blurred_img = blurred_img.astype(np.uint8)
        _, binary_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self.contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return self.contours

    def resize_and_pad(self, img, size, shrink):
        h, w = img.shape
        target_digit_size = int(size * shrink)
        scale = target_digit_size / max(h, w)
        resized = cv2.resize(img, (int(w * scale), int(h * scale)))
        h_pad, w_pad = size - resized.shape[0], size - resized.shape[1]
        top, bottom = h_pad // 2, h_pad - h_pad // 2
        left, right = w_pad // 2, w_pad - w_pad // 2
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value = self.pad_value)
        return padded

    def extract_digits(self, shrink = 1.0, min_area = 9, min_dim = 3):
        _ = self.find_contours()
        digits = []
        self.locs = []
        for contour in self.contours:
            if cv2.contourArea(contour) < min_area:
                continue
            points = contour[:, 0, :]
            if points.size == 0:
                continue
            min_x, min_y = np.min(points, axis = 0)
            max_x, max_y = np.max(points, axis = 0)
            width, height = max_x - min_x, max_y - min_y
            if width < min_dim or height < min_dim:
                continue
            if max_x > min_x and max_y > min_y:
                digit_roi = self.I[min_y:max_y+1, min_x:max_x+1]
                blank = np.zeros_like(digit_roi)
                contour_shifted = contour - [min_x, min_y]
                cv2.drawContours(blank, [contour_shifted], -1, (255), thickness = 1)
                digit_resized = self.resize_and_pad(blank, size = self.size, shrink = shrink)
                digits.append(digit_resized)
                self.locs.append((min_x, min_y, max_x, max_y))
        self.digits = digits
        return np.array(digits)

    def show_contours(self):
        contour_img = self.image.copy()
        cv2.drawContours(contour_img, self.contours, -1, (0, 255, 0), 2)
        contour_img_rgb = cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize = (8, 8))
        plt.imshow(contour_img_rgb)
        plt.title("Visualization of Contours")
        plt.axis("off")
        plt.show()

    def show_recognized_digits(self, predictions = None):
        annotated = self.image.copy()
        labels = predictions
        if predictions is None:
            predictions = ["" for _ in range(len(self.digits))]
        for (min_x, min_y, max_x, max_y), label in zip(self.locs, predictions):
            cv2.rectangle(annotated, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
            cv2.putText(annotated, str(label), (min_x, min_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        if labels is None:
            plt.title("Recognized Digits")
        else:
            plt.title("Recognized Digits with Labels")
        plt.show()

    def show_extracted_digits(self):
        for i, d in enumerate(self.digits):
            plt.subplot(len(self.digits) // 5 + 1, 5, i + 1)
            plt.imshow(d, cmap = 'gray')
            plt.axis('off')
        plt.suptitle(f"Extracted Digits ({self.size}x{self.size})")
        plt.show()
