import numpy as np
import sys

from edge_detector import EdgeDetector
from shape_context import ShapeContext
from matcher import Matcher


def print_bar(count, total, index, size):
    percent = int((count / total) * 100)
    bar = '#' * (percent // 2) + ' ' * (50 - percent // 2)
    sys.stdout.write(f'\rProgress: [{bar}] {percent}% ({index + 1}/{size})')
    sys.stdout.flush()

class KNN:
    """ Shape-context-based KNN classifier """

    def __init__(self, k = 3, use_cv2 = True, max_points = 50):
        self.k = k
        self.use_cv2 = use_cv2
        self.max_pts = max_points
        self.train_sample_points = []
        self.train_descriptors = []
        self.train_images = []
        self.train_labels = []

    def get_points_and_descriptors(self, X):
        """ Extract sampled points and descriptors for a list of images """
        edge_points_list = [EdgeDetector(img, use_cv2 = self.use_cv2).detect_edges() for img in X]
        print("Edge Detection is done!")
        shape_contexts = [ShapeContext(pts, max_points = self.max_pts) for pts in edge_points_list]

        sampled_points = [sc.sampled_points for sc in shape_contexts]
        descriptors = [sc.compute_shape_context() for sc in shape_contexts]
        print("Computation of Descriptors is done!")
        print("-" * 75)

        return sampled_points, descriptors

    def fit(self, X_train, y_train):
        """ Store training data and precompute descriptors """
        self.train_images = X_train
        self.train_labels = y_train
        self.train_sample_points, self.train_descriptors = self.get_points_and_descriptors(X_train)

    def predict(self, X_test, alpha = 1.0, beta = 0.3, gamma = 1.6):
        """ Predict labels for test images by computing shape distances """
        test_sample_points, test_descriptors = self.get_points_and_descriptors(X_test)
        predictions = []
        n_train = len(self.train_images)
        n_test = len(X_test)

        print("Predicting...")
        for idx in range(n_test):
            test_img = X_test[idx]
            test_points = test_sample_points[idx]
            test_desc = test_descriptors[idx]

            costs = []
            for i in range(n_train):
                print_bar(i + 1, n_train, idx, n_test)
                matcher = Matcher(
                    self.train_images[i], test_img,
                    self.train_descriptors[i], test_desc,
                    self.train_sample_points[i], test_points
                )
                cost = matcher.total_cost(alpha, beta, gamma)
                costs.append(cost)

            print()
            best_indices = np.argpartition(costs, self.k)[:self.k]
            top_labels = self.train_labels[best_indices]
            label = np.bincount(top_labels).argmax()
            predictions.append(label)

        print("-" * 75)
        return np.array(predictions)

    def accuracy(self, y_pred, y_test):
        """ Evaluate the classifier with test data """
        return np.sum(y_pred == y_test) / len(y_test)
