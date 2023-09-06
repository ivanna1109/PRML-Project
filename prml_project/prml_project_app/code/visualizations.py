import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import pandas as pd
import datasets_load as dsl

def iris_correlation_matrix():
    X, y, feature_names = dsl.load_iris()
    correlation_matrix = pd.DataFrame(X, columns=feature_names).corr()
    sns.set(style='white')
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()

def iris_distribution():
    print("test")
    iris = datasets.load_iris()
    sepal_length = iris.data[:, 0]
    sepal_width = iris.data[:, 1]
    petal_length = iris.data[:, 2]
    petal_width = iris.data[:, 3]

    sns.set(style='whitegrid')
    plt.figure(figsize=(8, 5))

    sns.kdeplot(sepal_length, label='Sepal Length', fill=True)
    sns.kdeplot(sepal_width, label='Sepal Width', fill=True)
    sns.kdeplot(petal_length, label='Petal Length', fill=True)
    sns.kdeplot(petal_width, label='Petal Width', fill=True)

    plt.title('Distribution of attributes in Iris dataset')
    plt.xlabel('Attribute value (in cm)')
    plt.ylabel('Density')
    plt.legend()

    plt.show()

def iris_histogram():
    iris = datasets.load_iris()
    sepal_length = iris.data[:, 0]
    sepal_width = iris.data[:, 1]
    petal_length = iris.data[:, 2]
    petal_width = iris.data[:, 3]
    sns.set(style='whitegrid')
    plt.figure(figsize=(10, 6))
    sns.histplot(sepal_length, bins=20, label='Sepal Length', color='blue', alpha=0.7)
    sns.histplot(sepal_width, bins=20, label='Sepal Width', color='green', alpha=0.7)
    sns.histplot(petal_length, bins=20, label='Petal Length', color='red', alpha=0.7)
    sns.histplot(petal_width, bins=20, label='Petal Width', color='purple', alpha=0.7)
    plt.title('Histogram in Iris dataset')
    plt.xlabel('Attribute value (in cm)')
    plt.ylabel('Number of instances')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    #correlation_matrix()
    #iris_distribution()
    iris_histogram()