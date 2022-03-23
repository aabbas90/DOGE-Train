import csv, os
import gurobipy as gp

# input_data_dir = '/BS/discrete_opt/nobackup/miplib_collection/lp_format_presolved/'
# data_csv = csv.reader(open('/home/ahabbas/projects/learnDA/utils/bdd_solver_comparison_with_pc_3_8_22_mma_cuda.csv', 'r'))
# out_csv_path = '/home/ahabbas/projects/learnDA/utils/bdd_solver_comparison_with_pc_3_8_22_mma_cuda_with_unsolved.csv'
# data_names = []
# for (i, line) in enumerate(data_csv):
#     if i == 0:
#         continue
#     data_names.append(os.path.basename(line[0]))

# for disk_name in os.listdir(input_data_dir):
#     if disk_name in data_names or not disk_name.endswith('.lp'):
#         continue

#     print(disk_name)

# for path, subdirs, files in os.walk('/home/ahabbas/data/learnDBCA/miplib_crops'):
#     for instance_name in sorted(files):
#         if not instance_name.endswith('.lp'):
#             continue
        
#         instance_path = os.path.join(path, instance_name)
#         ilp_gurobi = gp.read(instance_path)
#         variables = ilp_gurobi.getVars()
#         invalid = True
#         for var in variables:
#             invalid = False
#             # GM can contain continuous variables even though they will ultimately have binary value. TODOAA.
#             if (var.VType != 'B'): #, f'Variable {var} is not binary in file {ilp_path} and instead of type {var.VType}'
#                invalid = True
#                break
#         if invalid:
#             os.remove(instance_path)
#             print(f'Removed: {instance_path}')


num = 0
for path, subdirs, files in os.walk('/home/ahabbas/data/learnDBCA/miplib_crops/open'):
    for instance_name in sorted(files):
        if not instance_name.endswith('.lp'):
            continue
        
        num += 1
        # instance_path = os.path.join(path, instance_name)
        # ilp_gurobi = gp.read(instance_path)
        # variables = ilp_gurobi.getVars()
        # invalid = True
        # for var in variables:
        #     invalid = False
        #     # GM can contain continuous variables even though they will ultimately have binary value. TODOAA.
        #     if (var.VType != 'B'): #, f'Variable {var} is not binary in file {ilp_path} and instead of type {var.VType}'
        #        invalid = True
        #        break
        # if invalid:
        #     os.remove(instance_path)
        #     print(f'Removed: {instance_path}')
print(num)