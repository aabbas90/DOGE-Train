import os, pickle
import time
import csv

suffix_simplex='_gurobi_dual-simplex_log.txt'
suffix_barrier='_gurobi_barrier_x16_log.txt'

ct_instances = {'ct': ['flywing_245']}
worms_instances = {'worm': [f'worm{i}-16-03-11-1745' for i in range(10, 31)]}
# Skipped 'chr25a', 'chr22b' due to no logs. Very fast. 
qaplib_small_instances = {'qaplib_small' : ['bur26h', 'had20', 'lipa40b', 'nug30', 'ste36c', 'tho30', 'bur26g', 'lipa40a', 'nug28', 'scr20', 'tai35b', 'kra32', 'nug27', 'rou20', 'tai35a']}
ind_set_instances = {'ind_set': ['10', '13', '16', '19', '21', '24', '27', '2', '32', '59', '61', '64', '67', '6', '72', '75', '78', '80', '83', '86', '11', '14', '17', '1', '22', '25', '28', '30', '3', '5', '62', '65', '68', '70', '73', '76', '79', '81', '84', '8', '12', '15', '18', '20', '23', '26', '29', '31', '4', '60', '63', '66', '69', '71', '74', '77', '7', '82', '85', '9']}
mrf_pf_instances = { 'mrf_pf': ['2BBN', '2BCX', '2F3Y', '2FOT', '2HQW', '2O60', '3BXL']}

sp_root_dir = '/BS/discrete_opt/nobackup/bdd_experiments/'
is_root_dir = '/home/ahabbas/data/learnDBCA/independent_set_random/test_logs/'
mrf_pf_root_dir = '/home/ahabbas/data/learnDBCA/cv_structure_pred/mrf/protein_folding/test_large_logs/'

import os
import re
import csv
import argparse
import numpy as np
from scipy.interpolate import interp1d

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
            dual_infeasibility = float(result[4])
            if len(result.groups()) == 5 and dual_infeasibility == 0.0:
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
                return np.array(collected_data), True
            result=re.match("\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*)\s+(\d+)s", line)
            if result is None:
                continue
            collected_data.append([float(result[7]), float(result[3])])
        if 'Iter' in line:
            started = True
    return np.array(collected_data), False

def find_file(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
    return None

def try_write_max_dual_obj(dataset_name, instance_name, current_max_obj, is_optimal):
    gt_dir = 'gt_objs'
    os.makedirs(gt_dir, exist_ok = True)
    gt_file = os.path.join(gt_dir, dataset_name + instance_name + '.txt')
    if os.path.isfile(gt_file):
        with open(gt_file, 'rb') as f:
            obj_data = pickle.load(f)
            existing_obj = obj_data['max_obj']
            existing_is_optimal = obj_data['is_optimal']
        if current_max_obj <= existing_obj or existing_is_optimal:
            return existing_obj
    with open(gt_file, 'wb') as f:
        pickle.dump({'max_obj': current_max_obj, 'is_optimal': is_optimal}, f)
        return current_max_obj

def try_get_min_dual_obj(dataset_name, instance_name, current_min_obj): # Minimum dual objective used to normalize relative gap.
    gt_dir = 'init_objs'
    os.makedirs(gt_dir, exist_ok = True)
    gt_file = os.path.join(gt_dir, dataset_name + instance_name + '.txt')
    if os.path.isfile(gt_file):
        with open(gt_file, 'rb') as f:
            existing_obj = np.load(f)
            return existing_obj
    print(f'Warning: Written initial lower bound to disk for instance: {instance_name}. This should be coming from BDD solver.')
    #breakpoint() # should be coming from BDD solver initial lb.
    with open(gt_file, 'wb') as f:
        np.save(f, current_min_obj)
    return current_min_obj

def clip_max(arr):
    return np.maximum.accumulate(arr)

def compute_relative_gaps(dual_obj_arr, min_dual_obj, max_dual_obj):
    return (max_dual_obj - dual_obj_arr)  / (max_dual_obj - min_dual_obj) 

def extrapolate_log(log_data, max_time, si):
    current_max_time = log_data[0, :].max()
    t_extra = np.arange(current_max_time + si, max_time + si, si)
    vals_extra = np.array([log_data[1, :].max(), log_data[2, :].min()]).reshape((2, 1)).repeat(axis = 1, repeats = len(t_extra))
    t_extra = t_extra[np.newaxis, :]
    extra = np.concatenate((t_extra, vals_extra), 0)
    return np.concatenate((log_data, extra), 1)

def merge_logs(acc_sum_log, instance_log, si):
    acc_max_time = acc_sum_log[0, :].max()
    instance_max_time = instance_log[0, :].max()
    if acc_max_time < instance_max_time:
        acc_sum_log = extrapolate_log(acc_sum_log, instance_max_time, si)
    elif acc_max_time > instance_max_time:
        instance_log = extrapolate_log(instance_log, acc_max_time, si)
    acc_sum_log[1, :] = acc_sum_log[1, :] + instance_log[1, :]
    acc_sum_log[2, :] = acc_sum_log[2, :] + instance_log[2, :]
    return acc_sum_log

def parse_dataset(instance_dict, root_dir, suffix, si = 5):
    max_time = 0
    acc_sum_log = None
    count = 0
    for dataset_name, instance_names in instance_dict.items():
        for instance_name_temp in instance_names:
            instance_name = instance_name_temp.strip()
            log_file_name = instance_name.lstrip().rstrip() + suffix
            log_file = find_file(log_file_name, root_dir)
            if log_file is None:
                print(f'Could not find log file by name {log_file_name} in {root_dir}')
                assert False
            log_data = open(log_file, "r").readlines()
            if suffix == suffix_simplex:
                current_log, is_optimal = get_log_simplex(log_data)
            elif suffix == suffix_barrier:
                current_log, is_optimal = get_log_barrier(log_data)
            else:
                assert(False)
            if (len(current_log) <= 1):
                print(f"Skipping: {instance_name}")
                continue
            max_dual_obj = try_write_max_dual_obj(dataset_name, instance_name, np.max(current_log[:, 1]).item(), is_optimal)
            min_dual_obj = try_get_min_dual_obj(dataset_name, instance_name, np.min(current_log[:, 1]).item())
            current_log[:, 1] = clip_max(current_log[:, 1])
            resampled, max_time = resample_log(current_log, si, max_time)
            resampled_rel_gaps = compute_relative_gaps(resampled[1:2, :], min_dual_obj, max_dual_obj)
            logs_instance = np.concatenate((resampled, resampled_rel_gaps), 0) # time, clipped dual obj, clipped relative dual gaps.
            if acc_sum_log is not None:
                acc_sum_log = merge_logs(acc_sum_log, logs_instance, si)
            else:
                acc_sum_log = logs_instance
            count = count + 1
        acc_sum_log[1, :] /= count
        acc_sum_log[2, :] /= count 
        # Now get rid of -infinity lb and greater than 1.0 relative gaps.
        valid_start_indices = acc_sum_log[2, :] <= 1.0
        valid_acc_avg_log = acc_sum_log[:, valid_start_indices]
        out_dir = dataset_name + '_logs'
        os.makedirs(out_dir, exist_ok = True)
        out_name = os.path.join(out_dir, os.path.splitext(suffix)[0][1:] + '.csv')
        f = open(out_name, 'w')
        writer = csv.writer(f, delimiter=',')
        for i in range(valid_acc_avg_log.shape[1]):
            writer.writerow([valid_acc_avg_log[0, i], valid_acc_avg_log[1, i], valid_acc_avg_log[2, i]])
        f.close()

#parse_dataset(ct_instances, sp_root_dir, suffix_simplex, si = 5)
#parse_dataset(worms_instances, sp_root_dir, suffix_simplex, si = 5)
#parse_dataset(qaplib_small_instances, sp_root_dir, suffix_simplex, si = 5)
#parse_dataset(qaplib_small_instances, sp_root_dir, suffix_barrier, si = 5)
parse_dataset(ind_set_instances, is_root_dir, suffix_simplex, si = 2)
#parse_dataset(mrf_pf_instances, mrf_pf_root_dir, suffix_simplex, si = 10)