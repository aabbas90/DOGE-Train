import os
import pickle

gt_dir='/BS/discrete_opt/nobackup/bdd_experiments/qaplib/'
root_dir='/home/ahabbas/data/learnDBCA/cv_structure_pred/qaplib_full/small/'

maps = {}
for path, subdirs, files in os.walk(gt_dir):
    for gt_name in sorted(files):
        if not gt_name.endswith('.txt'):
            continue
        
        gt_file = os.path.join(path, gt_name)
        with open(gt_file) as f:
            lines = f.readlines()
        for line in lines:
            if 'Root relaxation: objective' in line:
                obj = float(line.split('Root relaxation: objective ')[1].split(',')[0])
                time = float(line.split('iterations, ')[1].split()[0])
                instance_name = gt_name.split('_gurobi')[0]
                maps[instance_name] = [obj, time]
                break
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
        gt_info = pickle.load(open(sol_path, 'rb'))
        if instance_name in maps:
            ilp_stats = gt_info['ilp_stats'].copy()
            gt_info['lp_stats']['obj'] = float(maps[instance_name][0])
            gt_info['lp_stats']['time'] = float(maps[instance_name][1])
            gt_info['ilp_stats'] = ilp_stats
            ilp_obj = gt_info['ilp_stats']['obj']
            #print(f'{instance_name}: {maps[instance_name][0]}, {ilp_obj}')
        else:
            #print(f'{instance_name} not found')
            lp_stats = gt_info['lp_stats'].copy()
            lp_stats['obj_type'] = 'primal_obj'
            gt_info['lp_stats'] = lp_stats
            gt_info['ilp_stats'].pop('obj_type')   
        pickle.dump(gt_info, open(sol_path, "wb"))
