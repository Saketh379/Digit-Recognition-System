import numpy as np
from hog import HOG
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

class HOG_Classifier:

    def __init__(self, classifier = 'SVM', k = 3, C = 1.0, kernel = 'rbf', built_in = True):
        self.classifier = classifier
        self.k = k
        self.C = C
        self.kernel = kernel
        self.built_in = built_in

    def get_descriptors(self, img):
        return HOG(img, built_in = self.built_in).compute_descriptors()

    def fit(self, X_train, y_train):
        self.X = np.array([self.get_descriptors(img) for img in X_train])
        self.y = y_train
        if self.classifier == "SVM":
            self.model = SVC(C = self.C, kernel = self.kernel)
        elif self.classifier == "KNN":
            self.model = KNeighborsClassifier(n_neighbors = self.k)
        self.model.fit(self.X, self.y)

    def predict(self, X_test):
        test_desc = np.array([self.get_descriptors(img) for img in X_test])
        return self.model.predict(test_desc)
    
    def accuracy(self, y_test, y_pred):
        return np.sum(y_pred == y_test) / len(y_test)
