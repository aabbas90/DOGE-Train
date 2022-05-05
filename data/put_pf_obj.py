import os
import csv
import pickle

gt_file='pf_trws.csv'
root_dir='/home/ahabbas/data/learnDBCA/cv_structure_pred/mrf/protein_folding/'
csv_data=csv.reader(open(gt_file, 'r'))

maps = {}
instance_name = None
for line in csv_data:
    if 'value' in line[1]:
        instance_name = line[0]
        ub = float(line[2])
    if 'bound' in line[1]:
        lb = float(line[2])
    if 'runtime' in line[1]:
        time = float(line[2])
        maps[instance_name] = [lb, ub, time]
print(maps)

for path, subdirs, files in os.walk(root_dir):
    if not 'instances' in path:
        continue
    for instance_name in sorted(files):
        if not instance_name.endswith('.lp'):
            continue

        sol_name = instance_name.replace('.lp', '.pkl')
        sol_path = os.path.join(path.replace('instances', 'solutions'), sol_name)
        instance_name = instance_name.replace('.lp', '')
        if instance_name in maps:
            empty_sol = {'time': None, 'obj': None, 'sol_dict': None, 'sol': None}
            gt_info = pickle.load(open(sol_path, 'rb'))
            gt_info['ilp_stats'] = gt_info['ilp_stats'].copy()
            gt_info['lp_stats']['obj'] = float(maps[instance_name][0])
            gt_info['lp_stats']['time'] = float(maps[instance_name][2])
            gt_info['ilp_stats']['obj'] = float(maps[instance_name][1])
            gt_info['ilp_stats']['time'] = float(maps[instance_name][2])
            print(gt_info)
            pickle.dump(gt_info, open(sol_path, "wb"))
