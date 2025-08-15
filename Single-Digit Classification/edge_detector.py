import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter

class EdgeDetector:
    """ Implements edge detection using either a custom pipeline or OpenCV's Canny """

    def __init__(self, image, sigma = 1, low_threshold = 50, high_threshold = 100, use_cv2 = False):
        """ Initialize the detector with a grayscale image and thresholds """
        self.I = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) if image.ndim == 3 else image
        self.sigma = sigma
        self.low = low_threshold
        self.high = high_threshold
        self.edges = None
        self.edges_locs = None
        self.use_cv2 = use_cv2

    def gradient(self, direction = 'x'):
        """ Compute image gradient in x or y directions using Sobel Operators """
        if direction == 'x':
            kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        elif direction == 'y':
            kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        return convolve2d(self.I, kernel, mode = 'same')

    def compute_polar_coords(self):
        """ Compute magnitude and angle of gradient at each pixel after Gaussian Smoothing """
        Ix = self.gradient(direction = 'x')
        Iy = self.gradient(direction = 'y')
        Gx = gaussian_filter(Ix, sigma = self.sigma)
        Gy = gaussian_filter(Iy, sigma = self.sigma)
        mag = np.hypot(Gx, Gy)
        angle = (np.arctan2(Iy, Ix) * 180 / np.pi) % 180
        return mag, angle

    def NMS(self):
        """ Non-Maximum Suppression to thin out the edges based on gradient direction """
        mag, angle = self.compute_polar_coords()

        h, w = mag.shape
        angle = angle % 180
        Z = np.zeros((h, w), dtype = np.float32)

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                angle_val = angle[i, j]
                q, r = 255, 255

                # Check direction and interpolate accordingly
                if (0 <= angle_val < 22.5) or (157.5 <= angle_val < 180):
                    q, r = mag[i, j+1], mag[i, j-1]
                elif (22.5 <= angle_val < 67.5):
                    q, r = mag[i+1, j-1], mag[i-1, j+1]
                elif (67.5 <= angle_val < 112.5):
                    q, r = mag[i+1, j], mag[i-1, j]
                elif (112.5 <= angle_val < 157.5):
                    q, r = mag[i-1, j-1], mag[i+1, j+1]
                if mag[i, j] >= q and mag[i, j] >= r:
                    Z[i, j] = mag[i, j]

        return Z

    def threshold(self, weak, strong):
        """ Apply double thresholding to identify strong, weak, and non-edges """
        nms = self.NMS()
        res = np.zeros_like(nms, dtype = np.uint8)
        res[nms >= self.high] = strong
        res[(nms >= self.low) & (nms < self.high)] = weak
        return res

    def hysteresis(self, weak, strong):
        """ Track edge connectivity using hysteresis to suppress isolated weak edges """
        thresh = self.threshold(weak, strong)
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype = np.uint8)
        strong_mask = (thresh == strong).astype(np.uint8)
        changed = True

        while changed:
            dilated = cv2.dilate(strong_mask, kernel, iterations = 1)
            connected = (dilated == 1) & (thresh == weak)

            if np.any(connected):
                thresh[connected] = strong
                strong_mask = (thresh == strong).astype(np.uint8)
            else:
                changed = False

        thresh[thresh != strong] = 0
        return thresh

    def detect_edges(self, weak = 50, strong = 255):
        """ Perform edge detection """
        if self.use_cv2:
            edges_cv2 = cv2.Canny(self.I.astype(np.uint8), threshold1 = self.low, threshold2 = self.high)
            self.edges = (edges_cv2 > 0).astype(np.uint8)
            self.edges_locs = np.argwhere(self.edges > 0)
        else:
            self.edges = self.hysteresis(weak, strong)
            self.edges_locs = np.argwhere(self.edges == 255)
        return self.edges_locs if self.edges_locs.size != 0 else None

    def show_edges(self):
        """ Display the detected edges """
        plt.imshow(self.edges, cmap = 'gray')
        plt.title("Detected Edges")
        plt.axis('off')
        plt.show()
