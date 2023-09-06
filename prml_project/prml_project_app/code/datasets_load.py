from sklearn import datasets
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import numpy as np
import pandas as pd

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

def load_wines():
    pdf = pd.read_csv('wine.csv')
    print(pdf.dtypes)
    return pdf;

def load_titanic():
    titanic = sns.load_dataset('titanic')
    columns_for_drop = ['survived', 'deck', 'embark_town']
    titanic = pd.get_dummies(titanic, columns=['sex'], prefix=['sex'])
    titanic = pd.get_dummies(titanic, columns=['embarked'], prefix=['embarked'])
    titanic = pd.get_dummies(titanic, columns=['who'], prefix=['who'])
    titanic = pd.get_dummies(titanic, columns=['class'], prefix=['class'])
    titanic = pd.get_dummies(titanic, columns=['adult_male'], prefix=['adult_male'])
    titanic = pd.get_dummies(titanic, columns=['alive'], prefix=['alive'])
    titanic = pd.get_dummies(titanic, columns=['alone'], prefix=['alone'])
    titanic['alone_True'] = titanic['alone_True'].astype(int)
    titanic['alone_False'] = titanic['alone_False'].astype(int)
    titanic['sex_female'] = titanic['sex_female'].astype(int)
    titanic['sex_male'] = titanic['sex_male'].astype(int)
    titanic['embarked_C'] = titanic['embarked_C'].astype(int)
    titanic['embarked_Q'] = titanic['embarked_Q'].astype(int)
    titanic['embarked_S'] = titanic['embarked_S'].astype(int)
    titanic['who_child'] = titanic['who_child'].astype(int)
    titanic['who_man'] = titanic['who_man'].astype(int)
    titanic['who_woman'] = titanic['who_woman'].astype(int)
    titanic['class_First'] = titanic['class_First'].astype(int)
    titanic['class_Second'] = titanic['class_Second'].astype(int)
    titanic['class_Third'] = titanic['class_Third'].astype(int)
    titanic['adult_male_False'] = titanic['adult_male_False'].astype(int)
    titanic['adult_male_True'] = titanic['adult_male_True'].astype(int)
    titanic['alive_no'] = titanic['alive_no'].astype(int)
    titanic['alive_yes'] = titanic['alive_yes'].astype(int)
    print(titanic.shape)
    titanic['age'] = titanic['age'].fillna(titanic['age'].mean())
    print(titanic.isna().sum())
    X = titanic.drop(columns=columns_for_drop, axis=1).values  # X sadrži sve kolone osim 'survived'
    y = titanic['survived']  # y sadrži samo kolonu 'survived'
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    cross_val = [X, y]
    print(X_train.shape)
    print(X_test.shape)
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
    load_wines()