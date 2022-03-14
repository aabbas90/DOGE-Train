import os, pickle

root_dir = '/home/ahabbas/data/learnDBCA/cv_structure_pred/'
for path, subdirs, files in os.walk(root_dir):
    for instance_name in sorted(files):
        if not instance_name.endswith('.lp') or 'nan' in instance_name or 'normalized' in instance_name:
            continue

        instance_path = os.path.join(path, instance_name)
        sol_path = os.path.join(path.replace('instances', 'solutions'), instance_name.replace('.lp', '.pkl'))
        if not os.path.exists(sol_path):
            continue

        sol = pickle.load(open(sol_path, 'rb'))
        if sol['ilp_stats']['sol_dict'] is None:
            continue

        bdd_repr_conv_path = instance_path.replace('.lp', '_bdd_repr_dual_converged.pkl')
        bdd_repr = pickle.load(open(bdd_repr_conv_path, 'rb'))
        solver = pickle.loads(bdd_repr['solver_data'])
        bdd_lb = solver.lower_bound() / bdd_repr['obj_multiplier'] + bdd_repr['obj_offset']

        rel_path = os.path.relpath(instance_path, root_dir)
        lp_obj = sol['lp_stats']['obj']
        ilp_obj = sol['ilp_stats']['obj']
        rel_gap = 100.0* (ilp_obj - lp_obj) / abs(ilp_obj)
        rel_gap_bdd = 100.0* (lp_obj - bdd_lb) / abs(lp_obj)
        num_cons = bdd_repr['num_cons']
        print(f'file: {rel_path}, LP obj: {lp_obj:.3f}, ILP obj: {ilp_obj:.3f}, Percent rel. gap: {rel_gap:.3f}, BDD LB: {bdd_lb:.3f}, Percent BDD LB gap: {rel_gap_bdd:.3f}, Num cons: {num_cons}')