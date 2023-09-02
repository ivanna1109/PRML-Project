import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from cvxopt import matrix, solvers
    
class LinearSVM:
    def __init__(self):
        self.alpha = None
        self.w = None
        self.b = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.idx_mask = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        # Convert inputs to cvxopt format
        # Explanation of this rewriting can be found DODAJ_GDE
        P = matrix(np.outer(y, y) * (X @ X.T))
        q = matrix(-np.ones((num_samples, 1)))
        G = matrix(-np.eye(num_samples))
        h = matrix(np.zeros((num_samples, 1)))
        A = matrix(np.reshape(y, (1, -1)), tc='d') 
        b = matrix(np.zeros(1), tc='d')
        
        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)
    
        self.alpha = np.array(solution['x']).flatten()
        self.idx_mask = self.alpha > 1e-5
        self.support_vectors = X[self.idx_mask]
        self.support_vectors_labels = y[self.idx_mask]  
    
        self.w = np.sum((self.alpha[self.idx_mask][:, None] * self.support_vectors_labels[:, None]) * self.support_vectors, axis=0)
        self.b = np.mean(self.support_vectors_labels - np.dot(self.support_vectors, self.w))            
    
    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)
    
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
