from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd

def crossvalidation(model, X, y):
    k = 10
    crossval_scores = []

    for i in range(k):
        X_train, X_test, y_train, y_test = split_dataset(X, y, i, k)
    
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        crossval_scores.append(round(score,2))

    mean_score = np.mean(crossval_scores)
    print("Cross validation score:", crossval_scores)
    print("Cross validation mean score:", round(mean_score,5))


def split_dataset(X, y, curr_fold, num_folds):
    num_of_instances = len(X)
    num_of_instances_per_fold = num_of_instances // num_folds

    # Odredite indekse početka i kraja trenutnog folda
    start_index = curr_fold * num_of_instances_per_fold
    end_index = (curr_fold + 1) * num_of_instances_per_fold

    # Podelite podatke na trening i test skup
    X_test = X[start_index:end_index]
    y_test = y[start_index:end_index]

    # Konstruišite trening skup tako da isključite trenutni fold
    X_train = np.concatenate([X[:start_index], X[end_index:]], axis=0)
    y_train = np.concatenate([y[:start_index], y[end_index:]], axis=0)

    return X_train, X_test, y_train, y_test

def cross_val_scores(linear=None, linear_non_sep=None, kernel=None, built_in=None, crossval=None):
    if (linear != None):
        print('***** Linear:')
        crossvalidation(linear, crossval[0], crossval[1])
    if (linear_non_sep != None):
        print('***** NonSep:')
        crossvalidation(linear_non_sep, crossval[0], crossval[1])
    if (kernel != None):
        print('***** Kernel:')
        crossvalidation(kernel, crossval[0], crossval[1])
    if (built_in != None):
        print('***** Built-in Kernel SVM:')
        crossvalidation(built_in, crossval[0], crossval[1])
    print("-"*90)

def test_implementation(svm, X_test, y_test):
    accuracy = round(svm.accuracy(X_test, y_test),5)
    precision = round(svm.precision(X_test, y_test),5)
    recall = round(svm.recall(X_test, y_test), 5)
    f1 = round(svm.f1(X_test, y_test), 5)
    confusion_matrix = svm.confusion_matrix(X_test, y_test)
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'Confusion Matrix:\n {confusion_matrix}\n')
    df_cf = pd.DataFrame(confusion_matrix)
    html_table_cm = df_cf.to_html(classes='table table-bordered custom-table', escape=False)
    list_of_metrics = [accuracy, precision, recall, f1, html_table_cm]
    return list_of_metrics

def built_svm_metrics(y_test, y_pred_built_in_kernel_svm):
    accuracy = round(accuracy_score(y_test, y_pred_built_in_kernel_svm),5)
    precision = round(precision_score(y_test, y_pred_built_in_kernel_svm, average='macro'),5)
    recall = round(recall_score(y_test, y_pred_built_in_kernel_svm, average='macro'),5)
    f1 = round(f1_score(y_test, y_pred_built_in_kernel_svm, average='macro'),5)
    conf_matrix = confusion_matrix(y_test, y_pred_built_in_kernel_svm)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", confusion_matrix)
    print()
    df_cf = pd.DataFrame(conf_matrix)
    html_table_cm = df_cf.to_html(classes='table table-bordered custom-table', escape=False)
    list_of_metrics = [accuracy, precision, recall, f1, html_table_cm]
    return list_of_metrics
