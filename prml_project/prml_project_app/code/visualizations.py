import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import pandas as pd
import datasets_load as dsl
import numpy as np

#------------------------------------IRIS VISUALIZATIONS----------------------------------------
def iris_correlation_matrix():
    X, y, feature_names = dsl.load_iris_for_visualization()
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

    plt.title('Distribution of Iris dataset attributes')
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
    plt.title('Distribution of Iris Dataset Attributes (Histogram)')
    plt.xlabel('Attribute value (in cm)')
    plt.ylabel('Number of instances')
    plt.legend()
    plt.show()

def iris_pairplot():
    iris = datasets.load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['species'] = iris.target
    sns.pairplot(iris_df, hue='species')
    plt.suptitle('Pair Plot of Iris Dataset')
    plt.show()

#------------------------------------------------------WINE VISUALIZATIONS-----------------------------------
def wine_plot():
    wine_df = dsl.load_wine_for_visualization()
    avg_values_by_class = wine_df.groupby('quality').mean()
    avg_values_by_class.plot(kind='bar', figsize=(10, 6))
    plt.grid()
    plt.title('Average Attribute Values by Wine Quality')
    plt.xlabel('Wine Class')
    plt.ylabel('Average Value')
    plt.xticks(rotation=0)
    plt.yticks(np.arange(0, 51, 5))
    plt.legend(title='Attribute', loc='upper left')
    plt.show()

def wine_distribution(x_attribute,y_attribute ):   
    wine_df = dsl.load_wine_for_visualization()
    print(wine_df.shape)
    wine_df['quality'].replace({'bad': 0 , 'good': 1}, inplace=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(wine_df[x_attribute], wine_df[y_attribute], c=wine_df['quality'], cmap='viridis', alpha=0.7)
    plt.title(f'Scatter Plot: {x_attribute} vs. {y_attribute}')
    plt.xlabel(x_attribute)
    plt.ylabel(y_attribute)
    plt.colorbar(label='Class')
    plt.grid(True)
    plt.show()
   
def wine_graphCorellation():
    winePdf = dsl.load_wine_for_visualization()
    winePdf['quality'].replace({'bad': 0 , 'good': 1}, inplace=True)
    corr = winePdf.corr()
    plt.figure(figsize=(10,6)) 
    plt.xticks(rotation=40)
    sns.heatmap(corr, cmap='coolwarm', annot=True)
    plt.show()
  
def wine_distributionPlot():
    wines = dsl.load_wine_for_visualization()
    fig = plt.figure(figsize = (10,12))
    ax = fig.gca()
    wines.hist(ax = ax)
    plt.show()

def wine_count_plot():
    wines = dsl.load_wine_for_visualization()
    plt.figure(figsize=(8, 6))
    sns.countplot(data=wines, x='quality')    
    plt.grid()
    plt.title('Count of Good and Bad')
    plt.xlabel('Quality')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['No', 'Yes'])
    plt.show()
    
#-----------------------------------------TITANIC VISUALIZATION--------------------------------------
def titanic_plot1():
    titanic_df = dsl.load_titanic_for_visualization()
    selected_attributes = ['age']
    plt.figure(figsize=(12, 6))
    plt.grid()
    for attribute in selected_attributes:
        sns.histplot(data=titanic_df, x=attribute, bins=30, kde=True, label=attribute)

    plt.title('Histogram of Age Attribute')
    plt.xlabel('Attribute Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def titanic_plot2():
    titanic_df = dsl.load_titanic_for_visualization()
    plt.figure(figsize=(8, 6))
    sns.countplot(data=titanic_df, x='survived')    
    plt.grid()
    plt.title('Count of Survivors and Non-Survivors')
    plt.xlabel('Survived')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['No', 'Yes'])
    plt.show()

def titanic_plot3():
    titanic_df = dsl.load_titanic_for_visualization()
    plt.figure(figsize=(10, 6))
    plt.grid()
    sns.scatterplot(data=titanic_df, x='age', y='fare', hue='survived', palette='viridis', size='class', sizes=(20, 200))
    plt.title('Scatter Plot: Age vs. Fare (Colored by Survival)')
    plt.xlabel('Age')
    plt.ylabel('Fare')
    plt.show()

def titanic_plot4():
    titanic_df = dsl.load_titanic_for_visualization()
    plt.figure(figsize=(12, 10))
    plt.grid()
    my_pallete = ['#eb4034', '#2c45e8']
    sns.boxplot(data=titanic_df, x='class', y='fare', hue='sex', palette=my_pallete)
    plt.title('Box Plot: Fare by Class and Gender')
    plt.xlabel('Class')
    plt.ylabel('Fare')
    plt.show()

#-------------------------------------------DIGITS VISUALIZATION----------------------------------
def digits_plot1():
    digits = dsl.load_digits_for_vizualization()
    print(digits.data.shape)
    plt.hist(digits.target, bins=range(11), rwidth=0.8, align='left', color='purple')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.xticks(np.arange(0,10,1))
    plt.title('Histogram of Class Distribution in Digits Dataset')
    plt.show()

def digits_images():
    digits = dsl.load_digits_for_vizualization()
    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    for i in range(4):
        axes[i].imshow(digits.images[i], cmap=plt.cm.gray_r)
        axes[i].set_title(f'Label: {digits.target[i]}')
    plt.show()

def digits_corr():
    digits = dsl.load_digits_for_vizualization()
    X = digits.images.reshape((len(digits.images), -1))

    correlation_matrix = np.corrcoef(X.T)

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap of Pixel Values in Digits Dataset')
    plt.show()
    
if __name__ == '__main__':
    #iris_correlation_matrix()
    #iris_distribution()
    #iris_histogram()
    #iris_pairplot()
    #wine_graphCorellation()
    wine_count_plot()
    #wine_distributionPlot()
    #wine_distribution('free sulfur dioxide', 'total sulfur dioxide')
    #titanic_plot4()
    #digits_plot1()
    #digits_images()
    #digits_corr()