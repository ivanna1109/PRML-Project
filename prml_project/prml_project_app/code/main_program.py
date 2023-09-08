from . import svm_classification as svm_class
from . import web_functions as webf

def get_results(dataset, binary, multi):
    return get_results_for_dataset(dataset, binary, multi)
        

def get_results_for_dataset(dataset, binary, multi):
    list_of_impl = []
    list_of_conf_matrix = []
    list1_of_crossvals = []
    list2_of_crossvals = []
    if (binary =='Linear'):
        list_of_impl, list_of_conf_matrix, list1_of_crossvals, list2_of_crossvals = webf.binary_svm(dataset, linear=True)
        print(list_of_conf_matrix)
    elif (binary =='Linear Non Separable'):
        list_of_impl, list_of_conf_matrix, list1_of_crossvals, list2_of_crossvals = webf.binary_svm(dataset, linear_non_sep=True)
    elif (binary =='Kernel'):
        list_of_impl,list_of_conf_matrix, list1_of_crossvals, list2_of_crossvals = webf.binary_svm(dataset,kernel=True)
    if (multi == 'Linear Non Separable One VS One' ):
        list_i_tmp, list_conf_tmp, crossval_tmp1,  crossval_tmp2 = webf.multi_svm(dataset, linear=True)
        list_of_impl.extend(list_i_tmp)
        list_of_conf_matrix.extend(list_conf_tmp)
        list1_of_crossvals.extend(crossval_tmp1)
        list2_of_crossvals.extend(crossval_tmp2)
    elif (multi == 'Kernel One VS One'):
        list_i_tmp, list_conf_tmp, crossval_tmp1, crossval_tmp2 = webf.multi_svm(dataset, kernel=True)
        list_of_impl.extend(list_i_tmp)
        list_of_conf_matrix.extend(list_conf_tmp)
        list1_of_crossvals.extend(crossval_tmp1)
        list2_of_crossvals.extend(crossval_tmp2)
    return list_of_impl, list_of_conf_matrix, list1_of_crossvals, list2_of_crossvals

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