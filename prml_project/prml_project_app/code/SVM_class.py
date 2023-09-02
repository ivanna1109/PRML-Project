import numpy as np
from sklearn import datasets
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from cvxopt import matrix, solvers
#from sklearn.datasets import fetch_openml
    
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
        solution = solvers.qp(P, q, G, h, A, b)
    
        self.alpha = np.array(solution['x']).flatten()
        self.idx_mask = self.alpha > 1e-5
        self.support_vectors = X[self.idx_mask]
        self.support_vectors_labels = y[self.idx_mask]  
    
        self.w = np.sum((self.alpha[self.idx_mask][:, None] * self.support_vectors_labels[:, None]) * self.support_vectors, axis=0)
        self.b = np.mean(self.support_vectors_labels - np.dot(self.support_vectors, self.w))            
    
    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)
    
    def precision(self, X, y_true):
        y_pred = self.predict(X)
        return precision_score(y_true, y_pred)

    def recall(self, X, y_true):
        y_pred = self.predict(X)
        return recall_score(y_true, y_pred)

    def f1(self, X, y_true):
        y_pred = self.predict(X)
        return f1_score(y_true, y_pred)
    
class LinearSVM1:
    def __init__(self, C=1.0):
        self.C = C
        self.alpha = None
        self.w = None
        self.b = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.idx_mask = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        # Convert inputs to cvxopt format
        P = matrix(np.outer(y, y) * (X @ X.T))
        q = matrix(-np.ones((num_samples, 1)))
        G = matrix(np.vstack((-np.eye(num_samples), np.eye(num_samples))))
        h = matrix(np.hstack((np.zeros(num_samples), self.C * np.ones(num_samples))))
        A = matrix(np.reshape(y, (1, -1)), tc='d') 
        b = matrix(np.zeros(1), tc='d')
        solution = solvers.qp(P, q, G, h, A, b)
    
        self.alpha = np.array(solution['x']).flatten()
        self.idx_mask = self.alpha > 1e-5
        self.support_vectors = X[self.idx_mask]
        self.support_vectors_labels = y[self.idx_mask]  
    
        self.w = np.sum((self.alpha[self.idx_mask][:, None] * self.support_vectors_labels[:, None]) * self.support_vectors, axis=0)
        self.b = np.mean(self.support_vectors_labels - np.dot(self.support_vectors, self.w))            
    
    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)
    
    def precision(self, X, y_true):
        y_pred = self.predict(X)
        return precision_score(y_true, y_pred)

    def recall(self, X, y_true):
        y_pred = self.predict(X)
        return recall_score(y_true, y_pred)

    def f1(self, X, y_true):
        y_pred = self.predict(X)
        return f1_score(y_true, y_pred)

class KernelSVM:
    def __init__(self, kernel='linear', degree=3, C=1.0):
        self.kernel = kernel
        self.degree = degree
        self.C = C
        self.alpha = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.idx_mask = None
        self.b = None

    def polynomial_kernel(self, X):
        return (X @ X.T + 1) ** self.degree
    
    def linear_kernel(self, X):
        return X @ X.T

    def fit(self, X, y):
        num_samples, num_features = X.shape
        
        # choose the kernel function
        if self.kernel == 'linear':
            K = self.linear_kernel(X)
        elif self.kernel == 'polynomial':
            K = self.polynomial_kernel(X)
        else:
            raise ValueError(f"Unknown kernel function: {self.kernel}")
        
        # Convert inputs to cvxopt format
        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones((num_samples, 1)))
        G = matrix(np.vstack((-np.eye(num_samples), np.eye(num_samples))))
        h = matrix(np.hstack((np.zeros(num_samples), self.C * np.ones(num_samples))))
        A = matrix(np.reshape(y, (1, -1)), tc='d') 
        b = matrix(np.zeros(1), tc='d')

        # solve QP problem
        solution = solvers.qp(P, q, G, h, A, b)
    
        self.alpha = np.array(solution['x']).flatten()
        self.idx_mask = self.alpha > 1e-5
        self.support_vectors = X[self.idx_mask]
        self.support_vectors_labels = y[self.idx_mask]
        self.b = np.mean(self.support_vectors_labels - np.sum((self.alpha[self.idx_mask] * self.support_vectors_labels)[:, None] * K[self.idx_mask, self.idx_mask], axis=0))

    def predict(self, X):
        if self.kernel == 'linear':
            K = X @ self.support_vectors.T
        elif self.kernel == 'polynomial':
            K = (X @ self.support_vectors.T + 1) ** self.degree
        
        print("Shape of alpha: ", self.alpha[self.idx_mask].shape)
        print("Shape of support vector labels: ", self.support_vectors_labels.shape)
        print("Shape of K: ", K.shape)
        
        decision = np.sum((self.alpha[self.idx_mask] * self.support_vectors_labels)[:, None] * K.T, axis=0) + self.b
        return np.sign(decision)


    def precision(self, X, y_true):
        y_pred = self.predict(X)
        return precision_score(y_true, y_pred)

    def recall(self, X, y_true):
        y_pred = self.predict(X)
        return recall_score(y_true, y_pred)

    def f1(self, X, y_true):
        y_pred = self.predict(X)
        return f1_score(y_true, y_pred)
    
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

y = (y == 0).astype(int) * 2 - 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=1, stratify=y)

svm = LinearSVM()
svm1 = LinearSVM1()

svm.fit(X_train, y_train)
svm1.fit(X_train, y_train)

y_pred = svm.predict(X_test)
y_pred1 = svm1.predict(X_test)

precision = svm.precision(X_test, y_test)
precision1 = svm1.precision(X_test, y_test)

recall = svm.recall(X_test, y_test)
recall1 = svm1.recall(X_test, y_test)

f1 = svm.f1(X_test, y_test)
f11 = svm1.f1(X_test, y_test)


print(f'Accuracy: {np.mean(y_pred == y_test) == np.mean(y_pred1 == y_test)}')
print(f'Precision: {precision == precision1}')
print(f'Recall: {recall == recall1}')
print(f'F1 Score: {f1 == f11}')
