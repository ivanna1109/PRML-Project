import numpy as np
from cvxopt import matrix, solvers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from itertools import combinations

class KernelSVM:
    def __init__(self, kernel='linear', degree=3, C=1.0):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.alpha = None
        self.w = None
        self.b = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.idx_mask = None

    def kernel_function(self, x1, x2):
        #print(f"x1 shape: {x1.shape}, x2 shape: {x2.shape}")
        if self.kernel == 'linear':
            result = np.inner(x1, x2)
        elif self.kernel == 'polynomial':
            result = (np.inner(x1, x2) + 1)**self.degree
        else:
            raise ValueError(f"Unknown kernel function: {self.kernel}")
        #print(f"result shape: {result.shape}, result type: {type(result)}")
        return result     
       
    def fit(self, X, y):
        num_samples, num_features = X.shape
        # Initialize kernel matrix
        K = np.zeros((num_samples, num_samples))
        for i in range(num_samples):
            for j in range(num_samples):
                K[i,j] = self.kernel_function(X[i], X[j])
                
        # Convert inputs to cvxopt format
        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones((num_samples, 1)))
        G = matrix(np.vstack((-np.eye(num_samples), np.eye(num_samples))))
        h = matrix(np.hstack((np.zeros(num_samples), self.C * np.ones(num_samples))))
        A = matrix(np.reshape(y, (1, -1)), tc='d') 
        b = matrix(np.zeros(1), tc='d')
        
        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)
    
        self.alpha = np.array(solution['x']).flatten()
        self.idx_mask = self.alpha > 1e-5
        self.support_vectors = X[self.idx_mask]
        self.support_vectors_labels = y[self.idx_mask]  
    
        if self.kernel == 'linear':
            self.w = np.sum((self.alpha[self.idx_mask][:, None] * self.support_vectors_labels[:, None]) * self.support_vectors, axis=0)
            self.b = np.mean(self.support_vectors_labels - np.dot(self.support_vectors, self.w))            
        else:
            self.w = None
            self.b = 0  # initialize self.b to zero
            self.b = np.mean([yi - self.decision_function(xi) for xi, yi in zip(self.support_vectors, self.support_vectors_labels)])

    def decision_function(self, X):
        X = np.atleast_2d(X)  # ensures X is at least 2D
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            result = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                s = 0
                for a, sv_y, sv in zip(self.alpha[self.idx_mask], self.support_vectors_labels, self.support_vectors):
                    s += a * sv_y * self.kernel_function(X[i], sv)
                result[i] = s
            return result + self.b
        
    def predict(self, X):
        return np.sign(self.decision_function(X))
    
    def accuracy(self, X, y_true):
        y_pred = self.predict(X)
        return accuracy_score(y_true, y_pred)

    def precision(self, X, y_true):
        y_pred = self.predict(X)
        return precision_score(y_true, y_pred)

    def recall(self, X, y_true):
        y_pred = self.predict(X)
        return recall_score(y_true, y_pred)

    def f1(self, X, y_true):
        y_pred = self.predict(X)
        return f1_score(y_true, y_pred)

    def confusion_matrix(self, X, y_true):
       y_pred = self.predict(X)
       return confusion_matrix(y_true, y_pred)

class OneVsOneKernelSVM:
    def __init__(self, kernel='linear', degree=3, C=1.0):
        self.kernel = kernel
        self.degree = degree
        self.C = C
        self.classifiers = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c1, c2 in combinations(self.classes, 2):
            # Create binary labels for the current pair
            binary_y = np.where((y == c1) | (y == c2), y, y)
            binary_y = np.where(binary_y == c1, -1, binary_y)
            binary_y = np.where(binary_y == c2, 1, binary_y)
            
            # Only take the samples that belong to either c1 or c2
            binary_y = binary_y[(y == c1) | (y == c2)]
            binary_X = X[(y == c1) | (y == c2)]
            
            # Train SVM for the current pair
            svm = KernelSVM(kernel=self.kernel, degree=self.degree, C=self.C)
            svm.fit(binary_X, binary_y)
            
            # Store the trained SVM
            self.classifiers[(c1, c2)] = svm

    def predict(self, X):
        votes = []
        for x in X:
            vote_count = {}
            for (c1, c2), svm in self.classifiers.items():
                prediction = svm.predict(x.reshape(1, -1))
                predicted_class = c1 if prediction == -1 else c2
                if predicted_class not in vote_count:
                    vote_count[predicted_class] = 0
                vote_count[predicted_class] += 1
            votes.append(max(vote_count, key=vote_count.get))
        return np.array(votes)

    def accuracy(self, X, y_true):
        y_pred = self.predict(X)
        return accuracy_score(y_true, y_pred)

    def precision(self, X, y_true):
        y_pred = self.predict(X)
        return precision_score(y_true, y_pred, average='micro')

    def recall(self, X, y_true):
        y_pred = self.predict(X)
        return recall_score(y_true, y_pred, average='micro')

    def f1(self, X, y_true):
        y_pred = self.predict(X)
        return f1_score(y_true, y_pred, average='micro')

    def confusion_matrix(self, X, y_true):
       y_pred = self.predict(X)
       return confusion_matrix(y_true, y_pred, labels=self.classes)
