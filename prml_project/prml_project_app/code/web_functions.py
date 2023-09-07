from . import LinearSVM as lsvm
from . import LinearNonSepSVM as lnsvm
from .LinearNonSepSVM import OneVsOneSVM
from . import KernelSVM as ksvm
from .KernelSVM import OneVsOneKernelSVM
from . import datasets_load as dl
from . import metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn import svm

def iris_binary_svm(dataset = None, linear=False, linear_non_sep=False, kernel = False):
    linear_svm = None ; linear_non_sep_svm = None ;kernel_svm = None
    if (linear):
        linear_svm = lsvm.LinearSVM()
    if (linear_non_sep):
        linear_non_sep_svm = lnsvm.LinearNonSepSVM()
    if (kernel):
        kernel_svm = ksvm.KernelSVM(kernel='polynomial')
    built_in_kernel_svm = svm.SVC(C=1.0, kernel='poly', coef0=1, gamma=1, degree=3)

    X_train, X_test, y_train, y_test, crossval = dataset_loading(dataset, False)
   # print("\n******************************Crossvalidation for binary SVM*******************************\n")
    #metrics.cross_val_scores(linear_svm, linear_non_sep_svm, kernel_svm, built_in_kernel_svm, crossval)
    list_for_linear = None; list_for_non_sep = None; list_for_kernel = None; 
    if (linear):
        linear_svm.fit(X_train, y_train)
        print('***** Linear:')
        list_for_linear = metrics.test_implementation(linear_svm, X_test, y_test)
    if (linear_non_sep):
        linear_non_sep_svm.fit(X_train, y_train)
        print('***** NonSep:')
        list_for_non_sep = metrics.test_implementation(linear_non_sep_svm, X_test, y_test)
    if (kernel):
        kernel_svm.fit(X_train, y_train)
        print('***** Kernel:')
        list_for_kernel = metrics.test_implementation(kernel_svm, X_test, y_test)
    print('***** Built-in Kernel SVM:')
    built_in_kernel_svm.fit(X_train, y_train)
    y_pred_built_in_kernel_svm = built_in_kernel_svm.predict(X_test)
    list_for_built = metrics.built_svm_metrics(y_test, y_pred_built_in_kernel_svm)
    list_of_impl = []
    list_of_conf_matrix = []
    if (list_for_linear != None):
        linear_list = ['Linear', list_for_linear[0], list_for_linear[1], list_for_linear[2], list_for_linear[3]]
        list_of_impl.append(linear_list)
        list_of_conf_matrix.append(list_for_linear[4])
    if (list_for_non_sep != None):
        linear_non_list = ['Linear Non Separable', list_for_non_sep[0], list_for_non_sep[1], list_for_non_sep[2], list_for_non_sep[3]]
        list_of_impl.append(linear_non_list)
        list_of_conf_matrix.append(list_for_non_sep[4])
    if (list_for_kernel != None):
        kernel_list = ['Kernel', list_for_kernel[0], list_for_kernel[1], list_for_kernel[2], list_for_kernel[3]]
        list_of_impl.append(kernel_list)
        list_of_conf_matrix.append(list_for_kernel[4])
    list_of_impl.append(['Built-in Kernel SVM',  list_for_built[0], list_for_built[1], list_for_built[2], list_for_built[3]])
    list_of_conf_matrix.append(list_for_built[4])
    return list_of_impl, list_of_conf_matrix

def iris_multi_svm(dataset, linear=False, kernel = False):
    linear_non_sep_svm_one_vs_one = None; built_in_lin_one_vs_one = None; kernel_one_vs_one_svm = None; built_in_kernel_one_vs_one_svm = None
    if (linear):
        linear_non_sep_svm_one_vs_one = OneVsOneSVM()
        built_in_lin_one_vs_one = svm.SVC(C=1.0, kernel='linear')
    if (kernel):
        kernel_one_vs_one_svm = OneVsOneKernelSVM(kernel='polynomial')
        built_in_kernel_one_vs_one_svm = svm.SVC(C=1.0, kernel='poly', coef0=1, gamma=1, degree=3)

    X_train_m, X_test_m, y_train_m, y_test_m , crossval= dataset_loading(dataset, True)
    print("\n***************************Crossvalidation for multiclass SVM****************************\n")
    #metrics.cross_val_scores(linear_non_sep_svm_one_vs_one, built_in_lin_one_vs_one, kernel_one_vs_one_svm, built_in_kernel_one_vs_one_svm, crossval)

    if (linear):
        linear_non_sep_svm_one_vs_one.fit(X_train_m, y_train_m)
        built_in_lin_one_vs_one.fit(X_train_m, y_train_m)
        print('***** Linear NonSep Multiclass:')
        list_for_linear = metrics.test_implementation(linear_non_sep_svm_one_vs_one, X_test_m, y_test_m)
        y_pred_built_in_lin_one_vs_one_svm = linear_non_sep_svm_one_vs_one.predict(X_test_m)
        print('***** Built-in Linear One-vs-one SVM:')
        list_for_linear_built = metrics.built_svm_metrics(y_test_m, y_pred_built_in_lin_one_vs_one_svm)
    if (kernel):
        kernel_one_vs_one_svm.fit(X_train_m, y_train_m)
        built_in_kernel_one_vs_one_svm.fit(X_train_m, y_train_m)
        print('***** Kernel One-vs-one SVM:')
        list_for_kernel = metrics.test_implementation(kernel_one_vs_one_svm, X_test_m, y_test_m)
        print('***** Built-in Kernel One-vs-one SVM:')
        y_pred_built_in_kernel_one_vs_one_svm = kernel_one_vs_one_svm.predict(X_test_m)
        list_for_kernel_built = metrics.built_svm_metrics(y_test_m, y_pred_built_in_kernel_one_vs_one_svm)
    list_of_impl = []
    if (linear):
        list_of_impl.append(['Linear Non Separable Multiclass', list_for_linear[0], list_for_linear[1], list_for_linear[2], list_for_linear[3]])
        list_of_impl.append(['Built-in Linear One-vs-one SVM', list_for_linear_built[0], list_for_linear_built[1], list_for_linear_built[2], list_for_linear_built[3]])
    if (kernel):
        list_of_impl.append(['Kernel One-vs-one SVM', list_for_kernel[0], list_for_kernel[1], list_for_kernel[2], list_for_kernel[3]])
        list_of_impl.append(['Built-in Kernel One-vs-one SVM', list_for_kernel_built[0], list_for_kernel_built[1], list_for_kernel_built[2], list_for_kernel_built[3]])
    return list_of_impl

def dataset_loading(dataset, multi):
    if(dataset == 'Iris' and not multi):
        return dl.load_iris_binary()
    if (dataset == 'Iris' and multi):
        return dl.load_iris_multi()
    if (dataset == 'Wines' and not multi):
        return dl.load_wines_binary()
    if (dataset == 'Wines' and multi):
        return dl.load_wines_multi()
    if (dataset == 'Titanic' and not multi):
        return dl.load_titanic_binary()
    if (dataset == 'Titanic' and  multi):
        return dl.load_titanic_multi()
    if (dataset == 'Digits' and not multi):
        return dl.load_digits_binary()
    if (dataset == 'Digits' and multi):
        return dl.load_digits_multi()

if __name__ == '__main__':
    iris_binary_svm(True, True)
