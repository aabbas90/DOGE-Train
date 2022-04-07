from cmath import nan
import os, pickle, time

root_dir = '/home/ahabbas/data/learnDBCA/cv_structure_pred/full_inst_mrf/'
# root_dir = '/home/ahabbas/data/learnDBCA/SPP_OR_Lib/'
for path, subdirs, files in os.walk(root_dir):
    for instance_name in sorted(files):
        if not instance_name.endswith('.lp') or 'nan' in instance_name or 'normalized' in instance_name:
            continue
        # if 'slow_bdd' in instance_name:
        #     continue
        # if (not os.path.exists(bdd_repr_path)):
        #     continue

        instance_path = os.path.join(path, instance_name)
        sol_path = os.path.join(path.replace('instances', 'solutions'), instance_name.replace('.lp', '.pkl'))
        if not os.path.exists(sol_path):
            continue

        sol = pickle.load(open(sol_path, 'rb'))
        # if sol['ilp_stats']['sol_dict'] is None:
        #     continue

        bdd_repr_path = instance_path.replace('.lp', '_bdd_repr.pkl')
        bdd_repr = pickle.load(open(bdd_repr_path, 'rb'))
        solver = pickle.loads(bdd_repr['solver_data'])
        bdd_initial_lb = solver.lower_bound() / bdd_repr['obj_multiplier'] + bdd_repr['obj_offset']

        bdd_solved_lb = nan
        bdd_repr_conv_path = instance_path.replace('.lp', '_bdd_repr_dual_converged.pkl')
        if (os.path.exists(bdd_repr_conv_path)):
            bdd_repr_conv = pickle.load(open(bdd_repr_conv_path, 'rb'))
            solver = pickle.loads(bdd_repr_conv['solver_data'])
            bdd_solved_lb = solver.lower_bound() / bdd_repr_conv['obj_multiplier'] + bdd_repr_conv['obj_offset']

        rel_path = os.path.relpath(instance_path, root_dir)
        lp_obj = sol['lp_stats']['obj']
        ilp_obj = sol['ilp_stats']['obj']
        if ilp_obj is None:
            ilp_obj = nan 
        if lp_obj is None:
            lp_obj = nan 
        rel_gap = 100.0* (ilp_obj - lp_obj) / (1e-8 + abs(ilp_obj))
        rel_gap_bdd = 100.0* (lp_obj - bdd_solved_lb) / (1e-8 + abs(lp_obj))
        rel_gap_loss = 100.0 * (lp_obj - bdd_solved_lb) / (1e-8 + lp_obj - bdd_initial_lb)           
        # if rel_gap_loss < 1.0:
        #     new_path = instance_path.replace('.lp', '_dual_solved.lp')
        #     os.rename(instance_path, new_path)
        #     print(f'renamed {instance_path} to {new_path}, rel_gap_loss: {rel_gap_loss}')

        num_cons = bdd_repr['num_cons']

        # if num_cons == 1:
        #     new_path = instance_path.replace('.lp', '_one_con.lp')
        #     os.rename(instance_path, new_path)
        #     print(f'renamed {instance_path} to {new_path}')

        num_layes = bdd_repr['num_layers']
        st = time.time()
        solver.non_learned_iterations(0.5, 10, 0.0)
        en = time.time()
        print(f'file: {rel_path:30}, Per iteration time: {(en - st) / 10.0:.3f}, LP obj: {lp_obj:11.3f}, ILP obj: {ilp_obj:.3f}, Percent rel. gap: {rel_gap:.3f}, BDD LB: {bdd_solved_lb:.3f}, Percent BDD LB gap: {rel_gap_bdd:.3f}, Percent BDD Gap Loss: {rel_gap_loss:.3f}, Num cons: {num_cons}, Num BDD layers: {num_layes}')
