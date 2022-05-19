import os, time
import gurobipy as gp
from contextlib import redirect_stdout

# root_dir = '/home/ahabbas/data/learnDBCA/independent_set_random/'
# instance_folder = os.path.join(root_dir, 'test_split/instances/')

root_dir = '/home/ahabbas/data/learnDBCA/cv_structure_pred/qaplib_full/test_split_large_wil/'
# lipa, sko on volta5
# tai, wil on volta12
instance_folder = os.path.join(root_dir, 'instances')
logs_folder = os.path.join(root_dir, 'test_large_logs')
if not os.path.isdir(logs_folder):
    os.makedirs(logs_folder)

def generate_gt_gurobi(ilp_path, logs_output_file):
    if os.path.exists(logs_output_file):
        return
    ilp_gurobi = gp.read(ilp_path)
    lp_relaxation = ilp_gurobi.relax()
    lp_relaxation.Params.Method = 2 # 1 -> Dual simplex, 2 -> Barrier.
    lp_relaxation.Params.TimeLimit = 3 * 3600 # 3 hour.
    with open(logs_output_file, 'w') as f:
        with redirect_stdout(f):
            lp_relaxation.optimize()
    print(f'\n\t\t Solved {ilp_path} to {logs_output_file}')

for instance_name in os.listdir(instance_folder):
    if not instance_name.endswith('.lp'):
        continue
    ilp_path = os.path.join(instance_folder, instance_name)
    # log_path = os.path.join(logs_folder, os.path.splitext(instance_name)[0] + '_gurobi_dual-simplex_log.txt')
    log_path = os.path.join(logs_folder, os.path.splitext(instance_name)[0] + '_gurobi_barrier_x16_log.txt')
    generate_gt_gurobi(ilp_path, log_path)