from cmath import nan
import os, pickle, time

root_dir = '/home/ahabbas/data/learnDBCA/miplib_crops/'
for path, subdirs, files in os.walk(root_dir):
    for instance_name in sorted(files):
        if not instance_name.endswith('.lp') or 'nan' in instance_name or 'normalized' in instance_name:
            continue

        instance_path = os.path.join(path, instance_name)
        sol_path = os.path.join(path.replace('instances', 'solutions'), instance_name.replace('.lp', '.pkl'))
        if not os.path.exists(sol_path):
            continue

        sol = pickle.load(open(sol_path, 'rb'))
        # if sol['ilp_stats']['sol_dict'] is None:
        #     continue

        # bdd_repr_conv_path = instance_path.replace('.lp', '_bdd_repr_dual_converged.pkl')
        # if (not os.path.exists(bdd_repr_conv_path)):
        #     continue
        # bdd_repr = pickle.load(open(bdd_repr_conv_path, 'rb'))
        # file_size = os.path.getsize(bdd_repr_conv_path)

        bdd_repr_path = instance_path.replace('.lp', '_bdd_repr.pkl')
        if 'slow_bdd' in instance_name:
            continue
        if (not os.path.exists(bdd_repr_path)):
            continue
        bdd_repr = pickle.load(open(bdd_repr_path, 'rb'))
        file_size = os.path.getsize(bdd_repr_path)

        solver = pickle.loads(bdd_repr['solver_data'])
        bdd_lb = solver.lower_bound() / bdd_repr['obj_multiplier'] + bdd_repr['obj_offset']

        rel_path = os.path.relpath(instance_path, root_dir)
        lp_obj = sol['lp_stats']['obj']
        ilp_obj = sol['ilp_stats']['obj']
        if ilp_obj is None:
            ilp_obj = nan 
        rel_gap = 100.0* (ilp_obj - lp_obj) / (1e-8 + abs(ilp_obj))
        rel_gap_bdd = 100.0* (lp_obj - bdd_lb) / (1e-8 + abs(lp_obj))
        num_cons = bdd_repr['num_cons']
        # if num_cons == 1:
        #     new_path = instance_path.replace('.lp', '_one_con.lp')
        #     os.rename(instance_path, new_path)
        #     print(f'renamed {instance_path} to {new_path}')
        # num_layes = bdd_repr['num_layers']
        # print(f'file: {rel_path:60}, ', end='')
        # st = time.time()
        # solver.non_learned_iterations(0.5, 10, 0.0)
        # en = time.time()
        # #print(f'file: {rel_path:60}, LP obj: {lp_obj:11.3f}, ILP obj: {ilp_obj:.3f}, Percent rel. gap: {rel_gap:.3f}, BDD LB: {bdd_lb:11.3f}, Percent BDD LB gap: {rel_gap_bdd:.3f}, Num cons: {num_cons}, Num BDD layers: {num_layes}')
        # print(f'Per iteration time: {(en - st) / 10.0:.3f}, Num cons: {num_cons}, Num BDD layers: {num_layes}.')