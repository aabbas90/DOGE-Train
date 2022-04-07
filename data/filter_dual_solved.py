from cmath import nan
import os, pickle, time

root_dir = '/home/ahabbas/data/learnDBCA/miplib_crops/'
gap_thresh = 1e-3
skipped = 0
filtered = 0
for path, subdirs, files in os.walk(root_dir):
    for instance_name in sorted(files):
        if not instance_name.endswith('.lp') or 'nan' in instance_name or 'normalized' in instance_name:
            continue

        if 'slow_bdd' in instance_name or 'dual_solved' in instance_name:
            continue

        instance_path = os.path.join(path, instance_name)
        sol_path = os.path.join(path.replace('instances', 'solutions'), instance_name.replace('.lp', '.pkl'))
        if not os.path.exists(sol_path):
            continue

        sol = pickle.load(open(sol_path, 'rb'))
        bdd_repr_path = instance_path.replace('.lp', '_bdd_repr.pkl')
        if (not os.path.exists(bdd_repr_path)):
            continue

        bdd_repr = pickle.load(open(bdd_repr_path, 'rb'))
        solver = pickle.loads(bdd_repr['solver_data'])
        bdd_lb = solver.lower_bound() / bdd_repr['obj_multiplier'] + bdd_repr['obj_offset']

        lp_obj = sol['lp_stats']['obj']
        lp_obj_normalized = (lp_obj - bdd_repr['obj_offset']) * bdd_repr['obj_multiplier']
        gap = lp_obj_normalized - solver.lower_bound()

        # gap = lp_obj - bdd_lb
        if gap < gap_thresh:
            print(f'Filtering file: {instance_path} with gap: {gap}, lp_obj: {lp_obj}, bdd_lb: {bdd_lb}.')
            filtered += 1
            os.rename(instance_path, os.path.join(path, instance_name.replace('.lp', '_dual_solved.lp')))
        else:
            # print(f'Skipping file: {instance_path} with gap: {gap}, lp_obj: {lp_obj}, bdd_lb: {bdd_lb}.')
            skipped += 1

print(f'Filtered: {filtered} / {filtered + skipped}.')