import csv, os

input_data_dir = '/BS/discrete_opt/nobackup/miplib_collection/lp_format_presolved/'
data_csv = csv.reader(open('/home/ahabbas/projects/learnDA/utils/bdd_solver_comparison_with_pc_3_8_22_mma_cuda.csv', 'r'))
out_csv_path = '/home/ahabbas/projects/learnDA/utils/bdd_solver_comparison_with_pc_3_8_22_mma_cuda_with_unsolved.csv'
data_names = []
for (i, line) in enumerate(data_csv):
    if i == 0:
        continue
    data_names.append(os.path.basename(line[0]))

for disk_name in os.listdir(input_data_dir):
    if disk_name in data_names or not disk_name.endswith('.lp'):
        continue

    print(disk_name)

