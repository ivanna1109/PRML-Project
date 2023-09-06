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

def load_mnist():
    mnist = joblib.load('mnist_data.pkl')
    X_subset = mnist.data[:10000]
    y_subset = mnist.target[:10000]
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42, stratify=y_subset)
    X_train = X_train / 255
    X_test = X_test / 255
    print(X_train.shape)
    encoder = OneHotEncoder(sparse=False, categories='auto')
    y_train = encoder.fit_transform(np.array(y_train).reshape(-1, 1))
    y_test = encoder.fit_transform(np.array(y_test).reshape(-1, 1))
    return X_train, X_test, y_train, y_test, None

if __name__ == '__main__':
    load_mnist()