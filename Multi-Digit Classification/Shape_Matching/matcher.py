import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2
import matplotlib.pyplot as plt

class Matcher:
    """
    Computes shape matching between two images using:
    - Shape Context cost (SC)
    - Thin Plate Spline bending energy (TPS)
    - Local Intensity Appearance cost (LIA)
    """

    def __init__(self, image1, image2, desc1, desc2, points1, points2, patch_size = 11):
        """ Initialize matcher with two grayscale images, shape context descriptors, and keypoints """
        self.img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) if image1.ndim == 3 else image1
        self.img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) if image2.ndim == 3 else image2
        self.h1 = np.asarray(desc1) if desc1 is not None else None
        self.h2 = np.asarray(desc2) if desc2 is not None else None
        self.points1 = points1 if points1 is not None else np.array([])
        self.points2 = points2 if points2 is not None else np.array([])
        self.patch_size = patch_size
        self.matches = None

    def cost_matrix(self):
        """ Compute chi-squared cost matrix between shape context histograms """
        if self.h1 is None or self.h2 is None:
            return np.array([])
        h1_exp, h2_exp = self.h1[:, np.newaxis, :], self.h2[np.newaxis, :, :]
        return 0.5 * np.sum(((h1_exp - h2_exp) ** 2) / (h1_exp + h2_exp + 1e-8), axis = 2)

    def SC_cost(self):
        """ Compute shape context cost and find optimal correspondences using Hungarian algorithm """
        if self.h1 is None or self.h2 is None:
            return float('inf')
        matrix = self.cost_matrix()
        if matrix.size == 0:
            return float('inf')
        row_ind, col_ind = linear_sum_assignment(matrix)
        self.matches = np.array(list(zip(row_ind, col_ind)))
        return matrix[row_ind, col_ind].sum()
        
    def TPS_cost(self):
        """ Compute Thin Plate Spline (TPS) bending energy between matched points """
        if self.matches is None or len(self.matches) == 0 or len(self.points1) == 0 or len(self.points2) == 0:
            return 0.0
        source = self.points1[self.matches[:, 0]]
        target = self.points2[self.matches[:, 1]]
        n = len(source)
        diff = source[:, np.newaxis, :] - target[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis = -1))
        K = (distances ** 2) * (np.log(np.maximum(distances ** 2, 1e-10)))
        P = np.hstack([np.ones((n, 1)), source])
        A = np.block([
            [K, P],
            [P.T, np.zeros((3, 3))]
        ])  
        Y = np.vstack([target, np.zeros((3, 2))])
        try:
            coeffs = np.linalg.solve(A, Y)
            cost = np.trace(coeffs[:n].T @ K @ coeffs[:n])
        except np.linalg.LinAlgError:
            cost = float('inf')
        return cost

    def extract_patch(self, img, center):
        """ Extract square patch of pixels around a center point for appearance matching """
        half = self.patch_size // 2
        y, x = int(center[0]), int(center[1])
        h, w = img.shape
        x_min = max(x - half, 0)
        x_max = min(x + half + 1, w)
        y_min = max(y - half, 0)
        y_max = min(y + half + 1, h)
        patch = img[y_min:y_max, x_min:x_max]
        if patch.shape != (self.patch_size, self.patch_size):
            pad_h = self.patch_size - patch.shape[0]
            pad_w = self.patch_size - patch.shape[1]
            patch = np.pad(patch, ((0, pad_h), (0, pad_w)), mode = 'constant', constant_values = 0)
        return patch.flatten()

    def LIA_cost(self):
        """ Compute appearance cost between matched patches centered at corresponding points """
        if self.matches is None or len(self.matches) == 0:
            return 0.0
        total = 0
        for idx1, idx2 in self.matches:
            if idx1 >= len(self.points1) or idx2 >= len(self.points2):
                continue
            p1 = self.points1[idx1]
            p2 = self.points2[idx2]
            patch1 = self.extract_patch(self.img1, p1)
            patch2 = self.extract_patch(self.img2, p2)
            patch1 = patch1 - np.mean(patch1)
            patch2 = patch2 - np.mean(patch2)
            total += np.sum((patch1 - patch2) ** 2)
        return total / len(self.matches)

    def total_cost(self, alpha = 1.0, beta = 0.3, gamma = 1.6):
        """ Combine all costs (shape context, TPS, appearance) with weights """
        d_sc = self.SC_cost() if alpha != 0 else 0.0
        d_bc = self.TPS_cost() if beta != 0 else 0.0
        d_ac = self.LIA_cost() if gamma != 0 else 0.0
        return alpha * d_sc + beta * d_bc + gamma * d_ac

    def visualize(self):
        """ Visualize correspondences in a single image canvas """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(np.ones_like(self.img1) * 255, cmap='gray')
        if self.points1.size > 0:
            ax.scatter(self.points1[:, 1], self.points1[:, 0], c='blue', s=10, label='Source')
        if self.points2.size > 0:
            ax.scatter(self.points2[:, 1], self.points2[:, 0], c='green', s=10, label='Target')
        if self.matches is not None:
            for idx1, idx2 in self.matches:
                if idx1 >= len(self.points1) or idx2 >= len(self.points2):
                    continue
                p1 = self.points1[idx1]
                p2 = self.points2[idx2]
                ax.plot([p1[1], p2[1]], [p1[0], p2[0]], color='red', linewidth=0.5, alpha=0.6)
        ax.set_title("Matcher Class Correspondences")
        ax.legend()
        ax.axis('off')
        plt.tight_layout()
        plt.show()
