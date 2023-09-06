import svm_classification as svm_class
import svm_classification as svm_class

def get_results(dataset, binary, multi):
    pass

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