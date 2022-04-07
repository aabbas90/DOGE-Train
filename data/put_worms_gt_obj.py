from copyreg import pickle
import os
import csv
import pickle

gt_file='/home/ahabbas/data/learnDBCA/worms_gurobi.log'
sol_dir='/home/ahabbas/data/learnDBCA/cv_structure_pred/full_inst/solutions/'
csv_data=csv.reader(open(gt_file, 'r'))

maps = {}
for line in csv_data:
    instance_name=line[0]
    obj = float(line[1])
    time = float(line[2].replace(' seconds', ''))
    maps[instance_name.replace('-', '')] = [obj, time]
#print(maps)
for sol_file in sorted(os.listdir(sol_dir)):
    sol_path=os.path.join(sol_dir, sol_file)
    gt_info = pickle.load(open(sol_path, 'rb'))
    sol_key = sol_file[:6]
    # if sol_key in maps:
    #     gt_info['ilp_stats']['obj'] = maps[sol_key][0]
    #     gt_info['ilp_stats']['time'] = maps[sol_key][1]
    #     print(gt_info['ilp_stats']['obj'])
    #     print(gt_info['ilp_stats']['time'])
    #     pickle.dump(gt_info, open(sol_path, "wb"))
    if 'worm' in sol_file:
        obj = gt_info['ilp_stats']['obj']
        print(f'{sol_file}, {obj}')
