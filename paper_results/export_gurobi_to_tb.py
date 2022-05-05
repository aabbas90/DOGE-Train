import os, pickle, re, time
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from scipy.interpolate import interp1d

suffix='_gurobi_dual-simplex_log.txt'

ct_instances = {'ct': ['flywing_245']}
worms_instances = {'worm': [f'worm{i}-16-03-11-1745' for i in range(10, 31)]}
# Skipped 'chr25a', 'chr22b' due to no logs. Very fast. 
qaplib_small_instances = {'qaplib_small' : ['bur26h', 'had20', 'lipa40b', 'nug30', ' ste36c', 'tho30', 'bur26g', 'lipa40a', 'nug28', 'scr20', ' tai35b', 'kra32', 'nug27', 'rou20', 'tai35a']}
ind_set_instances = {'ind_set': ['10', '13', '16', '19', '21', '24', '27', '2', '32', '59', '61', '64', '67', '6', ' 72', '75', '78', '80', '83', '86', '11', '14', '17', '1', ' 22', '25', '28', '30', '3', ' 5', ' 62', '65', '68', '70', '73', '76', '79', '81', '84', '8', '12', '15', '18', '20', '23', '26', '29', '31', '4', ' 60', '63', '66', '69', '71', '74', '77', '7', ' 82', '85', '9']}
mrf_pf_instances = { 'mrf_pf': ['2BBN', '2BCX', '2F3Y', '2FOT', '2HQW', '2O60', '3BXL']}

sp_root_dir = '/BS/discrete_opt/nobackup/bdd_experiments/'
is_root_dir = '/home/ahabbas/data/learnDBCA/independent_set_random/test_logs/'
mrf_pf_root_dir = '/home/ahabbas/data/learnDBCA/cv_structure_pred/mrf/protein_folding/test_large_logs/'

mrf_out_dir = '/home/ahabbas/projects/LearnDBCA/out_dual/MRF_PF/nobackup/v_new/gurobi_logs/'
ct_out_dir = '/home/ahabbas/projects/LearnDBCA/out_dual/CT/nobackup/vf/gurobi_logs/'

def resample_log(collected_data, si, max_time, fill_mode = 'max'):
    Told = collected_data[:, 0]
    start_time = 0
    max_time = max(Told.max(), max_time)
    X = collected_data[:, 1]
    if fill_mode == 'max': # for lower bounds
        F = interp1d(Told, X, fill_value=X.max(), bounds_error=False) 
    elif fill_mode == 'min':
        F = interp1d(Told, X, fill_value=X.min(), bounds_error=False) 
    else:
        assert False

    Tnew = np.arange(0.0, max_time + si, si)
    Xnew = F(Tnew)
    sampled = np.stack((Tnew, Xnew))
    start_indices = sampled[0, :] <= Told.min()
    if fill_mode == 'max':
        sampled[1, start_indices] = np.NINF
    return sampled, max_time

def get_log_simplex(log_data):
    started=False
    collected_data = []
    for (i, line) in enumerate(log_data):
        if started:
            if 'Root relaxation: objective' in line or 'Solved in ' in line:
                return np.array(collected_data), True
            if 'Root relaxation: time limit' in line or 'Stopped in' in line:
                print('Timelimit')
                return np.array(collected_data), False
            result = re.match("\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)s", line)
            if result is None:
                continue
            if len(result.groups()) == 5:
                collected_data.append([float(result[5]), float(result[2])])            
        if 'Iteration    Objective       Primal Inf.    Dual Inf.      Time' in line:
            started = True
    # Check if it was solved very fast without logging:
    for (i, line) in enumerate(log_data):
        if 'Root relaxation: objective' in line:
            result = re.match("Root relaxation: objective (.*), (\d+) iterations, (.*) seconds\n", line)
            if result == None or result[1] == None or result[3] == None:
                breakpoint()
            collected_data.append([float(result[3]), float(result[1])])
            breakpoint() # this situation should not arise.
            return np.array(collected_data)
    breakpoint() # this situation should not arise
    return np.array(collected_data)

def get_log_barrier(log_data):
    started=False
    collected_data = []
    for (i, line) in enumerate(log_data):
        if started:
            if 'Root relaxation: objective' in line:
                return np.array(collected_data)
            result=re.match("\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*)\s+(\d+)s", line)
            if result is None:
                return np.array(collected_data)
            collected_data.append([float(result[7]), float(result[3])])
        if 'Iter' in line:
            started = True
    return np.array(collected_data)

def find_file(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
    return None

def clip_max(arr):
    return np.maximum.accumulate(arr)

def extrapolate_log(log_data, max_time, si):
    current_max_time = log_data[0, :].max()
    t_extra = np.arange(current_max_time + si, max_time + si, si)
    vals_extra = np.array([log_data[1, :].max(), log_data[2, :].min()]).reshape((2, 1)).repeat(axis = 1, repeats = len(t_extra))
    t_extra = t_extra[np.newaxis, :]
    extra = np.concatenate((t_extra, vals_extra), 0)
    return np.concatenate((log_data, extra), 1)

def parse_dataset(instance_dict, root_dir, out_dir, prefix, si = 5):
    max_time = 0
    acc_sum_log = None
    count = 0
    writer = SummaryWriter(log_dir = out_dir)
    for dataset_name, instance_names in instance_dict.items():
        for instance_name_temp in instance_names:
            instance_name = instance_name_temp.strip()
            log_file_name = instance_name.lstrip().rstrip() + suffix
            log_file = find_file(log_file_name, root_dir)
            if log_file is None:
                print(f'Could not find log file by name {log_file_name} in {root_dir}')
                assert False
            log_data = open(log_file, "r").readlines()
            current_log, is_optimal = get_log_simplex(log_data)
            if (len(current_log) <= 1):
                print(f"Skipping: {instance_name}")
                continue
            current_log[:, 1] = clip_max(current_log[:, 1])
            resampled, max_time = resample_log(current_log, si, max_time)
            start_time = time.time()
            for s in range(resampled.shape[1]):
                writer.add_scalars(f'{prefix}_{instance_name}.lp/lower_bounds_pred_clip_', {'gurobi': resampled[1, s]}, global_step = s, walltime = start_time + float(resampled[0, s]))

parse_dataset(ct_instances, sp_root_dir, ct_out_dir, 'test_CT_TEST', si = 10)
#parse_dataset(worms_instances, sp_root_dir, si = 5)
# parse_dataset(qaplib_small_instances, sp_root_dir, si = 5)
# parse_dataset(ind_set_instances, is_root_dir, si = 2)
# parse_dataset(mrf_pf_instances, mrf_pf_root_dir, mrf_out_dir, 'test_MRF_PF_TEST', si = 10)