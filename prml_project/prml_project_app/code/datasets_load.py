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

def load_wines_binary():
    wines = pd.read_csv('D:\LetnjiSemestar/PRML/ProjekatSVM/PRML-Project/prml_project/prml_project_app/code/wine.csv')
    wines['quality'].replace({'bad': 0 , 'good': 1}, inplace=True)
    X = wines.drop(["quality"], axis = 1).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(X.shape)
    y = wines["quality"].values
    y = (y == 0).astype(int) * 2 - 1
    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.25)
    crossval = [X, y]
    return X_train, X_test, y_train, y_test, crossval

def load_wines_multi():
    wines = pd.read_csv('D:\LetnjiSemestar/PRML/ProjekatSVM/PRML-Project/prml_project/prml_project_app/code/wine.csv')
    wines['quality'].replace({'bad': 0 , 'good': 1}, inplace=True)
    X = wines.drop(["quality"], axis = 1).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(X.shape)
    y = wines["quality"].values
    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.25)
    crossval = [X, y]
    return X_train, X_test, y_train, y_test, crossval

def tmp_loading_titanic():
    titanic = sns.load_dataset('titanic')
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
    return titanic

def load_titanic_binary():
    titanic = tmp_loading_titanic()
    columns_for_drop = ['survived', 'deck', 'embark_town']
    X = titanic.drop(columns=columns_for_drop, axis=1).values  # X sadrži sve kolone osim 'survived'
    y = titanic['survived']  # y sadrži samo kolonu 'survived'
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = (y == 0).astype(int) * 2 - 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    cross_val = [X, y]
    return X_train, X_test, y_train, y_test, cross_val

def load_titanic_multi():
    titanic = tmp_loading_titanic()
    columns_for_drop = ['survived', 'deck', 'embark_town']
    X = titanic.drop(columns=columns_for_drop, axis=1).values  # X sadrži sve kolone osim 'survived'
    y = titanic['survived']  # y sadrži samo kolonu 'survived'
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    cross_val = [X, y]
    return X_train, X_test, y_train, y_test, cross_val

def load_digits_binary():
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = (y == 0).astype(int) * 2 - 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    cross_val = [X, y]
    return X_train, X_test, y_train, y_test, cross_val

def load_digits_multi():
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    cross_val = [X, y]
    return X_train, X_test, y_train, y_test, cross_val


if __name__ == '__main__':
    pass
    #load_digits()