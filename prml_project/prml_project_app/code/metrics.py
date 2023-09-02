from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

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
    print("Cross validation mean score:", mean_score)

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

def cross_val_scores(linear, linear_non_sep, kernel, built_in, crossval):
    print('***** Linear:')
    crossvalidation(linear, crossval[0], crossval[1])
    print('***** NonSep:')
    crossvalidation(linear_non_sep, crossval[0], crossval[1])
    print('***** Kernel:')
    crossvalidation(kernel, crossval[0], crossval[1])
    print('***** Built-in Kernel SVM:')
    crossvalidation(built_in, crossval[0], crossval[1])
    print("-"*90)

def test_implementation(svm, X_test, y_test):
    print(f'Accuracy: {svm.accuracy(X_test, y_test)}')
    print(f'Precision: {svm.precision(X_test, y_test)}')
    print(f'Recall: {svm.recall(X_test, y_test)}')
    print(f'F1 Score: {svm.f1(X_test, y_test)}')
    print(f'Confusion Matrix:\n {svm.confusion_matrix(X_test, y_test)}\n')

def built_svm_metrics(y_test, y_pred_built_in_kernel_svm):
    print("Accuracy:", accuracy_score(y_test, y_pred_built_in_kernel_svm))
    print("Precision:", precision_score(y_test, y_pred_built_in_kernel_svm, average='macro'))
    print("Recall:", recall_score(y_test, y_pred_built_in_kernel_svm, average='macro'))
    print("F1 Score:", f1_score(y_test, y_pred_built_in_kernel_svm, average='macro'))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_built_in_kernel_svm))
    print()
