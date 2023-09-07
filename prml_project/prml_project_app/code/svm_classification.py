from . import LinearSVM
from . import LinearNonSepSVM
from .LinearNonSepSVM import OneVsOneSVM
from . import KernelSVM
from .KernelSVM import OneVsOneKernelSVM
from . import datasets_load as dl
from . import metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn import svm

def binary_svm():
    linear_svm = LinearSVM()
    linear_non_sep_svm = LinearNonSepSVM()
    kernel_svm = KernelSVM(kernel='polynomial')
    built_in_kernel_svm = svm.SVC(C=1.0, kernel='poly', coef0=1, gamma=1, degree=3) 

    X_train, X_test, y_train, y_test, crossval = dl.load_iris_binary()
    print("\n******************************Crossvalidation for binary SVM*******************************\n")
    metrics.cross_val_scores(linear_svm, linear_non_sep_svm, kernel_svm, built_in_kernel_svm, crossval)

    linear_svm.fit(X_train, y_train)
    linear_non_sep_svm.fit(X_train, y_train)
    #svm2.fit(X_train, y_train)
    built_in_kernel_svm.fit(X_train, y_train)
    kernel_svm.fit(X_train, y_train)

    print('***** Linear:')
    metrics.test_implementation(linear_svm, X_test, y_test)
    print('***** NonSep:')
    metrics.test_implementation(linear_non_sep_svm, X_test, y_test)
    print('***** Kernel:')
    metrics.test_implementation(kernel_svm, X_test, y_test)
    print('***** Built-in Kernel SVM:')
    y_pred_built_in_kernel_svm = built_in_kernel_svm.predict(X_test)
    metrics.built_svm_metrics(y_test, y_pred_built_in_kernel_svm)

def multiclass_svm():
    linear_non_sep_svm_one_vs_one = OneVsOneSVM()
    built_in_lin_one_vs_one = svm.SVC(C=1.0, kernel='linear')
    kernel_one_vs_one_svm = OneVsOneKernelSVM(kernel='polynomial')
    built_in_kernel_one_vs_one_svm = svm.SVC(C=1.0, kernel='poly', coef0=1, gamma=1, degree=3)

    X_train_m, X_test_m, y_train_m, y_test_m , crossval= dl.load_iris_multi()
    print("\n***************************Crossvalidation for multiclass SVM****************************\n")
    metrics.cross_val_scores(linear_non_sep_svm_one_vs_one, built_in_lin_one_vs_one, kernel_one_vs_one_svm, built_in_kernel_one_vs_one_svm, crossval)

    linear_non_sep_svm_one_vs_one.fit(X_train_m, y_train_m)
    built_in_lin_one_vs_one.fit(X_train_m, y_train_m)
    kernel_one_vs_one_svm.fit(X_train_m, y_train_m)
    built_in_kernel_one_vs_one_svm.fit(X_train_m, y_train_m)

    print('***** Linear NonSep Multiclass:')
    metrics.test_implementation(linear_non_sep_svm_one_vs_one, X_test_m, y_test_m)
    y_pred_built_in_lin_one_vs_one_svm = linear_non_sep_svm_one_vs_one.predict(X_test_m)
    print('***** Built-in Linear One-vs-one SVM:')
    metrics.built_svm_metrics(y_test_m, y_pred_built_in_lin_one_vs_one_svm)
    print('***** Kernel One-vs-one SVM:')
    metrics.test_implementation(kernel_one_vs_one_svm, X_test_m, y_test_m)
    print('***** Built-in Kernel One-vs-one SVM:')
    y_pred_built_in_kernel_one_vs_one_svm = kernel_one_vs_one_svm.predict(X_test_m)
    metrics.built_svm_metrics(y_test_m, y_pred_built_in_kernel_one_vs_one_svm)

def binary_svm_wines():
    linear_svm = LinearSVM()
    linear_non_sep_svm = LinearNonSepSVM()
    kernel_svm = KernelSVM(kernel='polynomial')
    built_in_kernel_svm = svm.SVC(C=1.0, kernel='poly', coef0=1, gamma=1, degree=3) 

    wines =dl.load_wines()
    wines['quality'].replace({'bad': 0 , 'good': 1}, inplace=True)
    print(wines.dtypes)
    X = wines.drop(["quality"], axis = 1).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = wines["quality"].values
    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.25)
    print(X_train.shape)
    print(y_train.shape)
    crossval = [X, y]
    print("\n******************************Crossvalidation for binary SVM*******************************\n")
    metrics.cross_val_scores(linear_svm, linear_non_sep_svm, kernel_svm, built_in_kernel_svm, crossval)
    """
    linear_svm.fit(X_train, y_train)
    #linear_non_sep_svm.fit(X_train, y_train)
    #svm2.fit(X_train, y_train)
    #built_in_kernel_svm.fit(X_train, y_train)
    #kernel_svm.fit(X_train, y_train)

    print('***** Linear:')
    metrics.test_implementation(linear_svm, X_test, y_test)
    print('***** NonSep:')
    metrics.test_implementation(linear_non_sep_svm, X_test, y_test)
    print('***** Kernel:')
    metrics.test_implementation(kernel_svm, X_test, y_test)
    print('***** Built-in Kernel SVM:')
    y_pred_built_in_kernel_svm = built_in_kernel_svm.predict(X_test)
    metrics.built_svm_metrics(y_test, y_pred_built_in_kernel_svm)
    """

def multiclass_svm_titanic():
    linear_non_sep_svm_one_vs_one = OneVsOneSVM()
    built_in_lin_one_vs_one = svm.SVC(C=1.0, kernel='linear')
    kernel_one_vs_one_svm = OneVsOneKernelSVM(kernel='polynomial')
    built_in_kernel_one_vs_one_svm = svm.SVC(C=1.0, kernel='poly', coef0=1, gamma=1, degree=3)
    X_train_m, X_test_m, y_train_m, y_test_m , crossval= dl.load_titanic()
    #print("\n***************************Crossvalidation for multiclass SVM****************************\n")
    #metrics.cross_val_scores(linear_non_sep_svm_one_vs_one, built_in_lin_one_vs_one, kernel_one_vs_one_svm, built_in_kernel_one_vs_one_svm, crossval)

    linear_non_sep_svm_one_vs_one.fit(X_train_m, y_train_m)
    """
    built_in_lin_one_vs_one.fit(X_train_m, y_train_m)
    kernel_one_vs_one_svm.fit(X_train_m, y_train_m)
    built_in_kernel_one_vs_one_svm.fit(X_train_m, y_train_m)
    """
    print('***** Linear NonSep Multiclass:')
    metrics.test_implementation(linear_non_sep_svm_one_vs_one, X_test_m, y_test_m)
    y_pred_built_in_lin_one_vs_one_svm = linear_non_sep_svm_one_vs_one.predict(X_test_m)
    """
    print('***** Built-in Linear One-vs-one SVM:')
    metrics.built_svm_metrics(y_test_m, y_pred_built_in_lin_one_vs_one_svm)
    print('***** Kernel One-vs-one SVM:')
    metrics.test_implementation(kernel_one_vs_one_svm, X_test_m, y_test_m)
    print('***** Built-in Kernel One-vs-one SVM:')
    y_pred_built_in_kernel_one_vs_one_svm = kernel_one_vs_one_svm.predict(X_test_m)
    metrics.built_svm_metrics(y_test_m, y_pred_built_in_kernel_one_vs_one_svm)
    """

    
def multiclass_svm_wines():
    linear_non_sep_svm_one_vs_one = OneVsOneSVM()
    built_in_lin_one_vs_one = svm.SVC(C=1.0, kernel='linear')
    kernel_one_vs_one_svm = OneVsOneKernelSVM(kernel='polynomial')
    built_in_kernel_one_vs_one_svm = svm.SVC(C=1.0, kernel='poly', coef0=1, gamma=1, degree=3)

    wines =dl.load_wines()
    wines['quality'].replace({'bad': 0 , 'good': 1}, inplace=True)
    X = wines.drop(["quality"], axis = 1).values
    y = wines["quality"].values
    X_train_m, X_test_m, y_train_m, y_test_m  = train_test_split(X, y, test_size=0.25)
    crossval = [X, y]
    #X_train_m, X_test_m, y_train_m, y_test_m , crossval= dl.load_iris_multi()
    #print("\n***************************Crossvalidation for multiclass SVM****************************\n")
    #metrics.cross_val_scores(linear_non_sep_svm_one_vs_one, built_in_lin_one_vs_one, kernel_one_vs_one_svm, built_in_kernel_one_vs_one_svm, crossval)

    linear_non_sep_svm_one_vs_one.fit(X_train_m, y_train_m)
    """
    built_in_lin_one_vs_one.fit(X_train_m, y_train_m)
    kernel_one_vs_one_svm.fit(X_train_m, y_train_m)
    built_in_kernel_one_vs_one_svm.fit(X_train_m, y_train_m)
    """
    print('***** Linear NonSep Multiclass:')
    metrics.test_implementation(linear_non_sep_svm_one_vs_one, X_test_m, y_test_m)
    y_pred_built_in_lin_one_vs_one_svm = linear_non_sep_svm_one_vs_one.predict(X_test_m)
    """
    print('***** Built-in Linear One-vs-one SVM:')
    metrics.built_svm_metrics(y_test_m, y_pred_built_in_lin_one_vs_one_svm)
    print('***** Kernel One-vs-one SVM:')
    metrics.test_implementation(kernel_one_vs_one_svm, X_test_m, y_test_m)
    print('***** Built-in Kernel One-vs-one SVM:')
    y_pred_built_in_kernel_one_vs_one_svm = kernel_one_vs_one_svm.predict(X_test_m)
    metrics.built_svm_metrics(y_test_m, y_pred_built_in_kernel_one_vs_one_svm)
    """

    