def crossval_binary(dataset,linear=False, linear_non_sep=False, kernel=False):
    with open('D:\LetnjiSemestar\PRML\ProjekatSVM\PRML-Project\prml_project\prml_project_app\code\crossvalidation_files\\'+dataset.lower()+'_crossval.txt', 'r') as iris:
        line_number = 0
        choosen_impl = None
        built_in_line = None
        for line in iris:
            line_number += 1
            if line_number == 1 and linear:
                choosen_impl = line
            if line_number == 2 and kernel:
                choosen_impl = line
            if line_number == 3 and linear_non_sep:
                choosen_impl = line
            if line_number == 4:
                built_in_line = line
        result_impl = choosen_impl.split('|')
        result_built = built_in_line.split('|')
        result1, result2 = return_results(result_impl, result_built)
        return result1, result2
    
def crossval_multi(dataset, linear_non_separable, kernel):
    with open('D:\LetnjiSemestar\PRML\ProjekatSVM\PRML-Project\prml_project\prml_project_app\code\crossvalidation_files\\'+dataset.lower()+'_crossval.txt', 'r') as iris:
        line_number = 0
        choosen_impl = None
        built_in_line = None
        for line in iris:
            line_number += 1
            if line_number == 5 and linear_non_separable:
                choosen_impl = line
            if line_number == 6 and linear_non_separable:
                built_in_line = line
            if line_number == 7 and kernel:
                choosen_impl = line
            if line_number == 8 and kernel:
                built_in_line = line
        result_impl = choosen_impl.split('|')
        result_built = built_in_line.split('|')
        result1, result2 = return_results(result_impl, result_built)
        return result1, result2

def return_results(line_impl, line_built):
    line_impl[1] = line_impl[1][1:-1]
    line_built[1] = line_built[1][1:-1]
    list1 = [float(line_impl[2]), float(line_built[2])]
    list2 = [list(map(float, line_impl[1].split(','))),list(map(float, line_built[1].split(','))) ]
    return list1, list2

if __name__ == '__main__':
    crossval_binary('Iris', True)