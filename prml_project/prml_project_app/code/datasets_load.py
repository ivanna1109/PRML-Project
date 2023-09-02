from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_iris_binary(): 
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = (y == 0).astype(int) * 2 - 1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=1, stratify=y)
    cross_val = [X, y]
    return X_train, X_test, y_train, y_test, cross_val

def load_iris_multi():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    cross_val = [X, y]
    return X_train, X_test, y_train, y_test, cross_val