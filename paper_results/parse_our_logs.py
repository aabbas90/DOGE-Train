import os, pickle, time, csv, re, argparse
import numpy as np
from scipy.interpolate import interp1d

suffix_learned='lower_bounds_learned:'
suffix_non_learned='lower_bounds_non_learned:'

ct_instances = {
    'name': 'ct',
    'instances': ['flywing_245'],
    'prefix': 'test_CT_TEST_'
}
worms_instances = {
    'name': 'worm',
    'instances': [f'worm{i}-16-03-11-1745' for i in range(10, 31)], 
    'prefix': 'test_WORM_TEST_'
}
qaplib_small_instances = { # Skipped 'chr25a', 'chr22b' due to no logs. Very fast. 
    'name': 'qaplib_small',
    'instances' : ['bur26h', 'had20', 'lipa40b', 'nug30', 'ste36c', 'tho30', 'bur26g', 'lipa40a', 'nug28', 'scr20', 'tai35b', 'kra32', 'nug27', 'rou20', 'tai35a'],
    'prefix': 'test_QAP_TEST_'
}
qaplib_large_instances = { # Skipped 'chr25a', 'chr22b' due to no logs. Very fast. 
    'name': 'qaplib_large_instances',
    'instances' : ['lipa50b', 'lipa50a', 'lipa60a', 'lipa60b', 'lipa70b', 'lipa70a', 'tai40a', 'tai40b', 'tai50a', 'tai50b', 'tai60a', 'tai60b', 'tai64c', 'sko42', 'sko49', 'sko56', 'sko64', 'wil50'],
    'prefix': 'test_QAP_TEST_'
}
qaplib_all_instances = { # Skipped 'chr25a', 'chr22b' due to no logs. Very fast. 
    'name': 'qaplib_all_instances',
    'instances' : ['bur26h', 'had20', 'lipa40b', 'nug30', 'ste36c', 'tho30', 'bur26g', 'lipa40a', 'nug28', 'scr20', 'tai35b', 'kra32', 'nug27', 'rou20', 'tai35a', 'lipa50b', 'lipa50a', 'lipa60a', 'lipa60b', 'lipa70b', 'lipa70a', 'tai40a', 'tai40b', 'tai50a', 'tai50b', 'tai60a', 'tai60b', 'tai64c', 'sko42', 'sko49', 'sko56', 'sko64', 'wil50'],
    'prefix': 'test_QAP_TEST_'
}
ind_set_instances = {
    'name': 'ind_set',
    'instances': ['10', '13', '16', '19', '21', '24', '27', '2', '32', '59', '61', '64', '67', '6', '72', '75', '78', '80', '83', '86', '11', '14', '17', '1', '22', '25', '28', '30', '3', '5', '62', '65', '68', '70', '73', '76', '79', '81', '84', '8', '12', '15', '18', '20', '23', '26', '29', '31', '4', '60', '63', '66', '69', '71', '74', '77', '7', '82', '85', '9'],
    'prefix': 'test_MIS_'
}
mrf_pf_instances = {
    'name': 'mrf_pf',
    'instances': ['2BBN', '2BCX', '2F3Y', '2FOT', '2HQW', '2O60', '3BXL'],
    'prefix': 'test_MRF_PF_TEST_'
}

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

def try_write_max_dual_obj(dataset_name, instance_name, current_max_obj, is_optimal = False, obj_multiplier = 1.0):
    gt_dir = 'gt_objs'
    os.makedirs(gt_dir, exist_ok = True)
    gt_file = os.path.join(gt_dir, dataset_name + instance_name + '.txt')
    if os.path.isfile(gt_file):
        with open(gt_file, 'rb') as f:
            obj_data = pickle.load(f)
            existing_obj = obj_data['max_obj'] * obj_multiplier
            existing_is_optimal = obj_data['is_optimal']
        if current_max_obj <= existing_obj or existing_is_optimal:
            return existing_obj, existing_is_optimal
    with open(gt_file, 'wb') as f:
        print(f'Overwriting GT obj for {instance_name}.txt: current_max_obj: {current_max_obj}')
        pickle.dump({'max_obj': current_max_obj * obj_multiplier, 'is_optimal': is_optimal}, f)
        return current_max_obj, is_optimal

def try_write_max_time(dataset_name, instance_name, current_max_time):
    gt_dir = 'max_times'
    os.makedirs(gt_dir, exist_ok = True)
    gt_file = os.path.join(gt_dir, dataset_name + instance_name + '.txt')
    existing_time = np.nan
    if os.path.isfile(gt_file):
        with open(gt_file, 'rb') as f:
            time_data = pickle.load(f)
            existing_time = time_data['max_time']
        return existing_time
    assert False, f'max time should always come from gurobi for {instance_name}.'
    with open(gt_file, 'wb') as f:
        print(f'Overwriting max time for {instance_name}.txt: existing: {existing_time}, current_max_time: {current_max_time}')
        pickle.dump({'max_time': current_max_time}, f)
        return current_max_time

def try_write_min_dual_obj(dataset_name, instance_name, current_min_obj, offset = 0.0, overwrite_min_lb = False): # Minimum dual objective used to normalize relative gap.
    gt_dir = 'init_objs'
    # offset is used to convert initial lower bound computed through minimization problem to be valid for maximimzation.
    os.makedirs(gt_dir, exist_ok = True)
    gt_file = os.path.join(gt_dir, dataset_name + instance_name + '.txt')
    if os.path.isfile(gt_file):
        with open(gt_file, 'rb') as f:
            existing_obj = np.load(f)
        if not overwrite_min_lb:
            return offset - existing_obj
    with open(gt_file, 'wb') as f:
        np.save(f, offset - current_min_obj)
    return current_min_obj

def clip_max(arr):
    return np.maximum.accumulate(arr)

def compute_relative_gaps(dual_obj_arr, min_dual_obj, max_dual_obj):
    return (max_dual_obj - dual_obj_arr)  / (max_dual_obj - min_dual_obj) 

def compute_relative_gaps_integral(times_orig, rel_gap_orig, start_time, end_time):
    assert np.all(np.diff(times_orig) >= 0), 'times should be sorted.'
    times = np.copy(times_orig)
    rel_gap = np.copy(rel_gap_orig)
    rel_gap[rel_gap > 1.0] = 1.0
    rel_gap[rel_gap == np.NINF] = 0.0
    assert np.min(rel_gap) >= 0.0
    if times[0] < start_time:
        start_index =  np.where(times >= start_time)[0][0]
        times = times[start_index:]
        rel_gap = rel_gap[start_index:]
    if times[-1] > end_time: # Remove entries after end_time:
        end_index =  np.where(times > end_time)[0][0]
        times = times[:end_index]
        rel_gap = rel_gap[:end_index]
    valid_region_integral = np.trapz(rel_gap, x = times)
    # Add starting region area:
    if start_time < times[0]:
        starting_area = 0.5 * (times[0] - start_time) * (1.0 + rel_gap[0])
        valid_region_integral = valid_region_integral + starting_area
    # Add ending region area:
    if end_time > times[-1]:
        ending_area = 0.5 * (end_time - times[-1]) * (rel_gap[-1] + rel_gap[-1])
        valid_region_integral = valid_region_integral + ending_area
    return valid_region_integral

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
    if acc_max_time - instance_max_time <= si:
        acc_sum_log = extrapolate_log(acc_sum_log, instance_max_time, si)
    elif acc_max_time - instance_max_time >= si:
        instance_log = extrapolate_log(instance_log, acc_max_time, si)
    try:            
        acc_sum_log[1, :] = acc_sum_log[1, :] + instance_log[1, :]
        acc_sum_log[2, :] = acc_sum_log[2, :] + instance_log[2, :]
    except:
        breakpoint()
    return acc_sum_log

def match_instance_name_to_lines(log_data, instances, prefix, suffix):
    instances_with_prefix = [prefix + inst + '.lp' for inst in instances]
    instances_to_locs = {}
    last_detected_instance = None
    for (i, line) in enumerate(log_data):
        if line.strip() in instances_with_prefix:
            inst_name = line.strip().replace(prefix, '').replace('.lp', '')
            last_detected_instance = inst_name
        if last_detected_instance is not None and line.strip() == suffix and last_detected_instance not in instances_to_locs:
            instances_to_locs[last_detected_instance] = i
    if len(instances_to_locs) == len(instances):
        return instances_to_locs
    else:
        print(sorted(instances))
        print(sorted(instances_to_locs.keys()))
        breakpoint()
    return None

def get_log_values(log_data, start_line, metric_prefix = 'pred_itr_'):
    collected_data = []
    for line in log_data[start_line + 1:]:
        line_strip = line.strip()
        if line_strip.startswith(metric_prefix):
            k, val = line_strip.split(':')
            _, wall_time = k.split('_time_')
            collected_data.append([float(wall_time), float(val)])
        elif line.startswith('\t \t '):
            continue
        else:
            break
    collected_data = np.array(collected_data)
    collected_data[:, 0] -= collected_data[0, 0]
    return collected_data

def parse_log(instance_dict, log_file_path, si, suffix, obj_multiplier = 1.0, overwrite_min_lb = False, diff_tol = 1e-7):
    assert(obj_multiplier == 1.0 or obj_multiplier == -1.0)
    max_time = 0
    acc_sum_log = None
    sum_best_obj = 0
    sum_best_times = 0
    rel_gap_int_sum = 0
    count = 0
    log_data = open(log_file_path, "r").readlines()
    instance_to_locs = match_instance_name_to_lines(log_data, instance_dict['instances'], instance_dict['prefix'], suffix)
    if instance_to_locs is None:
        print(f'Unable to parse {log_file_path}')
        return
    print(f'Parsing {log_file_path}')
    print(instance_to_locs)
    dataset_name = instance_dict['name']
    for instance_name, start_line in instance_to_locs.items():
        current_log = get_log_values(log_data, start_line, metric_prefix = 'pred_itr_')
        if (len(current_log) <= 1):
            print(f"Skipping: {instance_name}")
            continue
        max_dual_obj, max_dual_is_optimal = try_write_max_dual_obj(dataset_name, instance_name, np.max(current_log[:, 1]).item(), obj_multiplier = obj_multiplier)
        offset = 0.0
        if obj_multiplier == -1.0:
            offset = max_dual_obj
        min_dual_obj = try_write_min_dual_obj(dataset_name, instance_name, current_log[0, 1], offset = offset, overwrite_min_lb = overwrite_min_lb)
        current_log[:, 1] = clip_max(current_log[:, 1])
        if max_dual_is_optimal:
            current_log[:, 1] = np.minimum(current_log[:, 1], max_dual_obj)
        resampled, max_time = resample_log(current_log, si, max_time)
        resampled_rel_gaps = compute_relative_gaps(resampled[1:2, :], min_dual_obj, max_dual_obj)
        
        rel_gap_difference = resampled_rel_gaps[0, :-1] - resampled_rel_gaps[0, 1:]
        best_index = resampled.shape[1] - 1
        if min(rel_gap_difference) < diff_tol:
            best_index =  np.where(rel_gap_difference < diff_tol)[0][0]
        logs_instance = np.concatenate((resampled, resampled_rel_gaps), 0) # time, clipped dual obj, clipped relative dual gaps.

        max_instance_time = try_write_max_time(dataset_name, instance_name, np.max(logs_instance[0, :]).item())
        rel_gap_int = compute_relative_gaps_integral(logs_instance[0, :], logs_instance[2, :], si, max_instance_time)
        rel_gap_int_sum += rel_gap_int
        sum_best_times += logs_instance[0, best_index]
        sum_best_obj += logs_instance[1, best_index]
        if acc_sum_log is not None:
            acc_sum_log = merge_logs(acc_sum_log, logs_instance, si)
        else:
            acc_sum_log = logs_instance
        count = count + 1
    acc_sum_log[1, :] /= count
    acc_sum_log[2, :] /= count 
    sum_best_obj /= count
    sum_best_times /= count
    rel_gap_int_sum /= count
    # Now get rid of -infinity lb.
    valid_start_indices = acc_sum_log[2, :] <= 1.0
    valid_acc_avg_log = acc_sum_log[:, valid_start_indices]
    out_dir = dataset_name + '_logs'
    os.makedirs(out_dir, exist_ok = True)
    log_name = os.path.basename(log_file_path).replace('.out', '')
    out_path = os.path.join(out_dir, suffix.replace(':', '_') + log_name)
    if os.path.exists(out_path + '.out'):
        os.remove(out_path + '.out')
    os.symlink(os.path.abspath(log_file_path), out_path + '.out')
    f = open(out_path + '.csv', 'w')
    writer = csv.writer(f, delimiter=',')
    for i in range(valid_acc_avg_log.shape[1]):
        writer.writerow([valid_acc_avg_log[0, i], valid_acc_avg_log[1, i], valid_acc_avg_log[2, i], sum_best_obj, sum_best_times, rel_gap_int_sum])
    f.close()

def parse_dataset(instance_dict, root_dir_or_log, suffix, si = 5, overwrite_min_lb = False, obj_multiplier = 1.0, diff_tol = 1e-7):
    if os.path.isfile(root_dir_or_log):
        parse_log(instance_dict, root_dir_or_log, si, suffix, obj_multiplier, overwrite_min_lb, diff_tol)
    else:
        for path, subdirs, files in os.walk(root_dir_or_log): # parse all log file in the directory.
            for filename in sorted(files):
                if not filename.endswith('.out'):
                    continue
                log_file = os.path.join(path, filename)
                parse_log(instance_dict, log_file, si, suffix, obj_multiplier, overwrite_min_lb, diff_tol)

# parse_dataset(ct_instances, 
#             '/home/ahabbas/projects/LearnDBCA/out_dual/slurm_new/9282678.out', 
#             suffix_non_learned, si = 5, overwrite_min_lb = True)

# parse_dataset(ct_instances, 
#             '/home/ahabbas/projects/LearnDBCA/out_dual/slurm_new/9349918.out', 
#             suffix_learned, si = 5, overwrite_min_lb = False)

# parse_dataset(ct_instances, 
#             '/home/ahabbas/projects/LearnDBCA/out_dual/slurm_new/9350818.out', 
#             suffix_learned, si = 5, overwrite_min_lb = False)
#parse_dataset(ct_instances, '/BS/ahmed_projects/work/projects/LearnDBCA/out_dual/slurm_new/9379768.out', suffix_learned, si = 5, overwrite_min_lb = False)
# parse_dataset(ct_instances, '/BS/ahmed_projects/work/projects/LearnDBCA/out_dual/slurm_new/9410026.out', suffix_learned, si = 5, overwrite_min_lb = False) #v9_lstm_longer_wo_grad_subg_1_1_16_16_8_3_1_1_400_False_False_1e-3_False_2_True_True_0.0_False_False_False_True

# parse_dataset(mrf_pf_instances, 
# '/home/ahabbas/projects/LearnDBCA/out_dual/MRF_PF/nobackup/v_new/v1_1_1_16_16_8_1_5_5_50_True_True_5e-3_False_2_True_True_0.1_False/eval_5_50_1e-9/9226107.out',
# suffix_non_learned, si = 10, overwrite_min_lb = True)
# parse_dataset(mrf_pf_instances, 
# '/home/ahabbas/projects/LearnDBCA/out_dual/MRF_PF/nobackup/v_new/v3_1_1_16_16_8_1_1_1_100_True_True_5e-3_False_2_True_True_0.1_False/eval_5_50_1e-9/9261837.out',
# suffix_learned, si = 10, overwrite_min_lb = False)

# parse_dataset(worms_instances, '/home/ahabbas/projects/LearnDBCA/out_dual/slurm/9282554_eval.out', suffix_non_learned, si = 2.5, overwrite_min_lb = True, diff_tol = 0.0)
# parse_dataset(worms_instances, '/home/ahabbas/projects/LearnDBCA/out_dual/slurm/9282555_eval.out', suffix_learned, si = 2.5, overwrite_min_lb = False)
# parse_dataset(worms_instances, '/home/ahabbas/projects/LearnDBCA/out_dual/slurm_new/9342493.out', suffix_learned, si = 2.5, overwrite_min_lb = False) # full
# parse_dataset(worms_instances, '/home/ahabbas/projects/LearnDBCA/out_dual/slurm_new/9352269.out', suffix_learned, si = 2.5, overwrite_min_lb = False) # no alpha
# parse_dataset(worms_instances, '/home/ahabbas/projects/LearnDBCA/out_dual/slurm_new/9352271.out', suffix_learned, si = 2.5, overwrite_min_lb = False) # no omega
# parse_dataset(worms_instances, '/home/ahabbas/projects/LearnDBCA/out_dual/slurm_new/9352274.out', suffix_learned, si = 2.5, overwrite_min_lb = False) # wo free update.
# parse_dataset(worms_instances, '/home/ahabbas/projects/LearnDBCA/out_dual/slurm_new/9352276.out', suffix_learned, si = 2.5, overwrite_min_lb = False) # only free update.
# parse_dataset(worms_instances, '/home/ahabbas/projects/LearnDBCA/out_dual/slurm_new/9353402.out', suffix_learned, si = 2.5, overwrite_min_lb = False) # no GNN
# parse_dataset(worms_instances, '/home/ahabbas/projects/LearnDBCA/out_dual/slurm_new/9415093.out', suffix_learned, si = 2.5, overwrite_min_lb = False) # lstm
# parse_dataset(worms_instances, '/home/ahabbas/projects/LearnDBCA/out_dual/slurm_new/9415253.out', suffix_learned, si = 2.5, overwrite_min_lb = False) # lstm last
# parse_dataset(worms_instances, '/BS/ahmed_projects/work/projects/LearnDBCA/out_dual/slurm_new/9415914.out', suffix_learned, si = 2.5, overwrite_min_lb = False) # wo lstm but rand start.
#parse_dataset(worms_instances, '/BS/ahmed_projects/work/projects/LearnDBCA/out_dual/slurm_new/9415908.out', suffix_learned, si = 2.5, overwrite_min_lb = False) # lstm rand start.

 
# parse_dataset(qaplib_small_instances, 
#             '/home/ahabbas/projects/LearnDBCA/out_dual/slurm_new/9296407.out',
#             suffix_non_learned,
#             si = 5, 
#             overwrite_min_lb = True)

# parse_dataset(qaplib_small_instances, 
#             '/home/ahabbas/projects/LearnDBCA/out_dual/slurm_new/9342494.out',
#             suffix_learned,
#             si = 5, 
#             overwrite_min_lb = False,
#             diff_tol = 1e-4)

# parse_dataset(qaplib_small_instances, 
# '/home/ahabbas/projects/LearnDBCA/out_dual/QAPLIB/nobackup/v_new2/v4_lstm_long_sep_1_1_16_16_8_3_5_5_500_True_True_1e-3_False_2_True_True_True_10_0.0_False_True/double_prec_epoch_404_20_5000_1e-9/9393391.out',
#             suffix_learned,
#             si = 5, 
#             overwrite_min_lb = False,
#             diff_tol = 1e-4)

# parse_dataset(qaplib_all_instances, 
#             '/BS/ahmed_projects/work/projects/LearnDBCA/paper_results/qaplib_non_learned_merged.out',
#             suffix_non_learned,
#             si = 5, 
#             overwrite_min_lb = True,
#             diff_tol = 1e-4)

# parse_dataset(qaplib_all_instances, 
#             '/home/ahabbas/projects/LearnDBCA/out_dual/QAPLIB/nobackup/v_new2/v4_lstm_long_sep_1_1_16_16_8_3_5_5_500_True_True_1e-3_False_2_True_True_True_10_0.0_False_True/last_v0_merged.out',
#             suffix_learned,
#             si = 5, 
#             overwrite_min_lb = False,
#             diff_tol = 1e-4)

# parse_dataset(qaplib_all_instances, 
#             '/home/ahabbas/projects/LearnDBCA/paper_results/qaplib_doge_merged.out',
#             suffix_learned,
#             si = 5, 
#             overwrite_min_lb = False,
#             diff_tol = 1e-4)

parse_dataset(ind_set_instances, 
            '/home/ahabbas/projects/LearnDBCA/out_dual/MIS/nobackup/vf/v6_mixed_prec_1_1_16_16_8_1_20_20_20_True_True_1e-3_False_1_True_True_0.0/double_prec_50_200_1e-9/9287189.out', 
            suffix_non_learned,
            si = 0.1, 
            overwrite_min_lb = True, 
            obj_multiplier = -1.0,
            diff_tol = 1e-15)

parse_dataset(ind_set_instances, 
            '/home/ahabbas/projects/LearnDBCA/out_dual/MIS/nobackup/vf/v6_mixed_prec_1_1_16_16_8_1_20_20_20_True_True_1e-3_False_1_True_True_0.0/double_prec_50_200_1e-9/9287189.out', 
            suffix_learned,
            si = 0.1, 
            overwrite_min_lb = False,
            obj_multiplier = -1.0)

parse_dataset(ind_set_instances, 
            '/home/ahabbas/projects/LearnDBCA/out_dual/slurm_new/9349914.out', # '/home/ahabbas/projects/LearnDBCA/out_dual/slurm_new/9349914.out', 
            suffix_learned,
            si = 0.1, 
            overwrite_min_lb = False,
            obj_multiplier = -1.0, 
            diff_tol = 0.0)

parse_dataset(ind_set_instances, 
            '/home/ahabbas/projects/LearnDBCA/out_dual/slurm_new/9431069.out', 
            suffix_learned,
            si = 0.1, 
            overwrite_min_lb = False,
            obj_multiplier = -1.0, 
            diff_tol = 0.0)
