from copyreg import pickle
import os
import csv
import pickle

root_dir='/home/ahabbas/data/learnDBCA/independent_set_random/test_split/'

for path, subdirs, files in os.walk(root_dir):
    for instance_name in sorted(files):
        if not instance_name.endswith('.lp'):
            continue
            
        instance_path = os.path.join(path, instance_name)
        bdd_repr_path = instance_path.replace('.lp', '_bdd_repr.pkl')
        sol_name = instance_name.replace('.lp', '.pkl')
        sol_path = os.path.join(path.replace('instances', 'solutions'), sol_name)

        bdd_repr = pickle.load(open(bdd_repr_path, 'rb'))
        gt_info = pickle.load(open(sol_path, 'rb'))

        bdd_repr['obj_multiplier'] = -1.0 * bdd_repr['obj_multiplier']
        assert(bdd_repr['obj_offset'] == 0.0)
        gt_info['lp_stats']['obj'] = -1.0 * gt_info['lp_stats']['obj']

        pickle.dump(bdd_repr, open(bdd_repr_path, "wb"), protocol = pickle.HIGHEST_PROTOCOL)
        pickle.dump(gt_info, open(sol_path, "wb"))
