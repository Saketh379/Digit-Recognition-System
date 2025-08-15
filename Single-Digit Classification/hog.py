import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.feature import hog as built_in_hog

class HOG:

    def __init__(self, image, cell_size = 4, block_size = 2, bins = 9, sigma = 1.0, built_in = True):
        self.I = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) if image.ndim == 3 else image.astype(np.float32)
        self.cs = cell_size
        self.bs = block_size
        self.bins = bins
        self.sigma = sigma
        self.built_in = built_in
        self.h, self.w = self.I.shape
        self.nx_cells, self.ny_cells = self.w // self.cs, self.h // self.cs

    def compute_polar_coords(self):
        smoothed = gaussian_filter(self.I, sigma = self.sigma)
        gx = np.gradient(smoothed, axis = 1)
        gy = np.gradient(smoothed, axis = 0)
        mag = np.hypot(gx, gy)
        angle = (np.arctan2(gy, gx) * 180 / np.pi) % 180
        return mag, angle

    def compute_cell_histograms(self, mag, angle):
        bin_size = 180 / self.bins
        bin_idx = np.floor(angle / bin_size).astype(int)
        bin_idx = np.clip(bin_idx, 0, self.bins - 1)

        hog = np.zeros((self.ny_cells, self.nx_cells, self.bins), dtype = np.float32)

        for i in range(self.ny_cells):
            for j in range(self.nx_cells):
                y0, y1 = i*self.cs, (i+1)*self.cs
                x0, x1 = j*self.cs, (j+1)*self.cs

                cell_mag = mag[y0:y1, x0:x1].reshape(-1)
                cell_bin = bin_idx[y0:y1, x0:x1].reshape(-1)

                hist = np.bincount(cell_bin, weights = cell_mag, minlength = self.bins)
                hog[i, j, :] = hist

        return hog

    def compute_descriptors(self):
        if self.built_in:
            features, hog_image = built_in_hog(
                self.I,
                orientations = self.bins,
                pixels_per_cell = (self.cs, self.cs),
                cells_per_block = (self.bs, self.bs),
                block_norm = 'L2-Hys',
                visualize = True,
                feature_vector = True
            )
            self.hog_features = hog_image
            return features

        mag, angle = self.compute_polar_coords()
        cell_hist = self.compute_cell_histograms(mag, angle)

        n_blocks_y = self.ny_cells - self.bs + 1
        n_blocks_x = self.nx_cells - self.bs + 1
        block_vector_len = self.bs * self.bs * self.bins
        features = np.empty((n_blocks_y * n_blocks_x, block_vector_len), dtype = np.float32)

        idx = 0
        for i in range(n_blocks_y):
            for j in range(n_blocks_x):
                block = cell_hist[i:i+self.bs, j:j+self.bs, :].flatten()
                block = np.clip(block, 0, 0.2)
                norm = np.linalg.norm(block) + 1e-6
                features[idx] = block / norm
                idx += 1

        return features.flatten()

    def visualize(self):
        plt.figure(figsize = (8, 8))
        plt.imshow(self.hog_features, cmap = 'gray')
        plt.axis('off')
        plt.title('HOG Visualization')
        plt.show()
