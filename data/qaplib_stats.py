from cmath import nan
import os, pickle, time
import gurobipy as gp

# train_dir = '/home/ahabbas/data/learnDBCA/cv_structure_pred/full_inst_qaplib/train_split/instances'
# test_dir = '/home/ahabbas/data/learnDBCA/cv_structure_pred/full_inst_qaplib/test_split/instances'
# max_num_var_train = 1e6
# max_num_con_train = 1e5
train_dir = '/home/ahabbas/data/learnDBCA/cv_structure_pred/full_inst_qaplib/test_split/instances'
test_dir = '/home/ahabbas/data/learnDBCA/cv_structure_pred/full_inst_qaplib/test_split_large/instances'
max_num_var_train = 1e6
max_num_con_train = 1e5
p = ''
for path, subdirs, files in os.walk(train_dir):
    for instance_name in sorted(files):
        if not instance_name.endswith('.lp'):
            continue

        instance_path = os.path.join(path, instance_name)
        ilp_gurobi = gp.read(instance_path)
        variables = ilp_gurobi.getVars()
        cons = ilp_gurobi.getConstrs()

        num_vars = len(variables)
        num_cons = len(cons)
        if num_vars > max_num_var_train or num_cons > max_num_con_train:
            new_path = os.path.join(test_dir, instance_name)
            os.rename(instance_path, new_path)
            bdd_instance_name = instance_name.replace('.lp', '_bdd_repr.pkl')
            bdd_repr_path = os.path.join(path, bdd_instance_name)
            if os.path.exists(bdd_repr_path):
                new_path = os.path.join(test_dir, bdd_instance_name)
                os.rename(bdd_repr_path, new_path)
        else:
            p = p + f'file: {instance_name:10}, Num vars: {num_vars:10}, Num cons: {num_cons:10}\n'

print(p)