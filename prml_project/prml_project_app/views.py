from django.shortcuts import render
import code.main_program as mp

def index_page(request):
    datasets = ['Iris', 'Wines', 'TMP2']
    binary_svms = ['Linear', 'Kernel', 'Linear Non Separable']
    multiclass_svms = ['Kernel', 'Linear Non Separable']
    context = {'datasets': datasets, 'binary_svms': binary_svms, 'multiclass_svms': multiclass_svms}
    return render(request, 'index.html', context)

def get_results(request):
    dataset = request.POST.get('dsname')
    binary = request.POST.get('binarySVM')
    multi = request.POST.get('multiclassSVM')
    results = mp.get_results(dataset, binary, multi)
    context = {'results':results}
    return render(request, 'index.html', context)




