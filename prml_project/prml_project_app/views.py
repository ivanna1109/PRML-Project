from django.shortcuts import render
from .code import main_program as mp
import sys

def index_page(request):
    print(sys.path)
    datasets = ['Iris', 'Wines', 'Titanic', 'Digits']
    binary_svms = ['Linear', 'Kernel', 'Linear Non Separable']
    multiclass_svms = ['Kernel', 'Linear Non Separable']
    context = {'datasets': datasets, 'binary_svms': binary_svms, 'multiclass_svms': multiclass_svms}
    return render(request, 'index.html', context)

def get_results(request):
    dataset = request.POST.get('dsname')
    binary = request.POST.get('binarySVM')
    multi = request.POST.get('multiclassSVM')
    results = mp.get_results(dataset, binary, multi)
    print(results)
    message = False
    if not results:
        message = 'You have to choose dataset and type of SVM implementation to see metrics!'
    datasets = ['Iris', 'Wines', 'Titanic', 'Digits']
    binary_svms = ['Linear', 'Kernel', 'Linear Non Separable']
    multiclass_svms = ['Kernel', 'Linear Non Separable']
    
    context = {'results':results, 'datasets': datasets, 'binary_svms': binary_svms, 'multiclass_svms': multiclass_svms, 'message': message}
    return render(request, 'index.html', context)




