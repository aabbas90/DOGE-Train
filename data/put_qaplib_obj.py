import os
import pickle

gt_file='/home/ahabbas/projects/LearnDBCA/data/qaplib_sol.txt'
root_dir='/home/ahabbas/data/learnDBCA/cv_structure_pred/full_inst_qaplib/small/train_split'
with open(gt_file) as f:
    lines = f.readlines()
maps = {}
for (i, line) in enumerate(lines):
    if i == 0:
        continue
    splits = line.split()
    name = splits[0]
    obj = splits[2]
    if 'OPT' in line:
        maps[name.lower()] = [obj, obj]
    elif '%' in line:
        lb = splits[4]
        maps[name.lower()] = [obj, lb]

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
            gt_info = {"lp_stats": empty_sol, "ilp_stats": empty_sol}
            # gt_info = pickle.load(open(sol_path, 'rb'))
            gt_info['ilp_stats']['obj'] = float(maps[instance_name][0])
            gt_info['lp_stats']['obj'] = float(maps[instance_name][1])
            #gt_info['ilp_stats']['time'] = maps[sol_key][1]
            print(gt_info['ilp_stats']['obj'])
            print(gt_info['lp_stats']['obj'])
            pickle.dump(gt_info, open(sol_path, "wb"))
