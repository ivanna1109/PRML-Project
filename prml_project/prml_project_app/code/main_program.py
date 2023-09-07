from . import svm_classification as svm_class
from . import web_functions as webf

def get_results(dataset, binary, multi):
    return get_results_for_dataset(dataset, binary, multi)
        

def get_results_for_dataset(dataset, binary, multi):
    list_of_impl = []
    if (binary =='Linear'):
        list_of_impl = webf.iris_binary_svm(dataset, linear=True)
        print(list_of_impl)
    elif (binary =='Linear Non Separable'):
        list_of_impl = webf.iris_binary_svm(dataset, linear_non_sep=True)
    elif (binary =='Kernel'):
        list_of_impl = webf.iris_binary_svm(dataset,kernel=True)
    if (multi == 'Linear Non Separable'):
        list_of_impl.extend(webf.iris_multi_svm(dataset, linear=True))
    elif (multi == 'Kernel'):
        list_of_impl.extend(webf.iris_multi_svm(dataset, kernel=True))
    return list_of_impl

def main():
    print("\n****************************************Binary SVM*****************************************\n")
    #svm_class.binary_svm()
    #svm_class.binary_svm_wines()
    #svm_class.binary_svm_titanic()
    #print("\n***********************************Multiclass SVM**************************************\n")
    #svm_class.multiclass_svm()
    #svm_class.multiclass_svm_wines() #radi ovo za multiclass, kernel se nešto oteze, sporo se racuna
    svm_class.multiclass_svm_titanic() #radi za titanik, sa binary ne radi (kao ni wina)

if __name__ == '__main__':
    main()