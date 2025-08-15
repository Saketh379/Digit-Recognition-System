import numpy as np
import matplotlib.pyplot as plt

class ShapeContext:
    """ Computes the shape context descriptor for a set of 2D points """

    def __init__(self, points, r_bins = 5, theta_bins = 12, r_inner = 0.125, r_outer = 2.0, max_points = 50):
        """
        Initialize the shape context descriptor.
        - points: Locations of detected pixels using Edge Detector.
        - r_bins: Number of bins for log-polar radius.
        - theta_bins: Number of angular bins.
        - r_inner: Inner radius fraction for binning.
        - r_outer: Outer radius multiple.
        - max_points: Maximum number of points to sample from input.
        """
        self.points = np.array(points) if points is not None else None
        self.r_bins = r_bins
        self.theta_bins = theta_bins
        self.r_inner = r_inner
        self.r_outer = r_outer
        self.max_pts = max_points
        self.sampled_points = None
        self.epsilon = 1e-6 # Small value to avoid division by zero

    def sample_points(self, pts):
        """ Uniformly sample a fixed number of points for computational efficiency """
        if len(pts) <= self.max_pts:
            return pts
        indices = np.linspace(0, len(pts) - 1, self.max_pts, dtype = int)
        return pts[indices]

    def compute_shape_context(self):
        """ Compute the shape context descriptor for each point """
        if self.points is None:
            return None
    
        pts = self.sample_points(self.points)
        self.sampled_points = pts
        n = len(pts)
        
        # Compute pairwise distances and angles
        dx = pts[:, np.newaxis, 0] - pts[np.newaxis, :, 0]
        dy = pts[:, np.newaxis, 1] - pts[np.newaxis, :, 1]
        r_array = np.sqrt(dx ** 2 + dy ** 2)
        theta_array = (np.arctan2(dy, dx) + 2 * np.pi) % (2 * np.pi)

        mean_dist = np.mean(r_array[r_array > 0])
        r_array /= (mean_dist + self.epsilon) # Normalize distances

        self.r_array = r_array
        self.theta_array = theta_array
        self.mean_dist = mean_dist

        # Define bin edges
        r_bin_edges = np.logspace(np.log10(self.r_inner), np.log10(self.r_outer), self.r_bins + 1)
        theta_bin_edges = np.linspace(0, 2 * np.pi, self.theta_bins + 1)

        histograms = []

        # For each point, compute histogram of relative positions
        for i in range(n):
            hist = np.zeros((self.r_bins, self.theta_bins))
            mask = np.ones(n, dtype = bool)
            mask[i] = False
            r = r_array[i, mask]
            theta = theta_array[i, mask]

            r_bins_inds = np.searchsorted(r_bin_edges, r, side = 'right') - 1
            theta_bins_inds = np.searchsorted(theta_bin_edges, theta, side = 'right') - 1
            
            # Keep only valid bins
            valid = (r_bins_inds >= 0) & (r_bins_inds < self.r_bins) & (theta_bins_inds >= 0) & (theta_bins_inds < self.theta_bins)
            for rb, tb in zip(r_bins_inds[valid], theta_bins_inds[valid]):
                hist[rb, tb] += 1
            histograms.append(hist.flatten())

        self.histograms = np.array(histograms)
        return self.histograms

    def visualize(self, index = 0):
        """ Visualize the shape context histogram and spatial bins for a given point """
        if self.points is None:
            return None
        pts = self.sampled_points
        y0, x0 = pts[index]
        r_bins = np.logspace(np.log10(self.r_inner), np.log10(self.r_outer), self.r_bins + 1)
        theta_bins = np.linspace(0, 2 * np.pi, self.theta_bins + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6))

        # Left: Log-polar bins visualization
        ax1.plot(pts[:, 1], pts[:, 0], 'o', label = 'Sampled Points')
        ax1.plot(x0, y0, 'ro', label = 'Reference Point')
        for r in r_bins:
            circle = plt.Circle((x0, y0), radius = r * self.mean_dist, color = 'gray', fill = False, linestyle = '--')
            ax1.add_patch(circle)
        for t in theta_bins:
            x1 = x0 + self.r_outer * self.mean_dist * np.cos(t)
            y1 = y0 + self.r_outer * self.mean_dist * np.sin(t)
            ax1.plot([x0, x1], [y0, y1], 'gray', linestyle = '--', linewidth = 0.5)
        ax1.set_aspect('equal')
        ax1.set_title(f"Log-Polar Bins at Point {index}")
        ax1.legend()
        ax1.invert_yaxis()

        # Right: Histogram heatmap
        hist = self.histograms[index].reshape(self.r_bins, self.theta_bins)
        im = ax2.imshow(hist, cmap = 'viridis', aspect = 'auto', interpolation = 'nearest')
        ax2.set_title(f"Shape Context Histogram (Index {index})")
        ax2.set_xlabel("Theta bins")
        ax2.set_ylabel("Log-Radial bins")
        ax2.invert_yaxis()
        fig.colorbar(im, ax = ax2, fraction = 0.046, pad = 0.04)

        plt.tight_layout()
        plt.show()
