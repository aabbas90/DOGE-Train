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
        print(f'Overwriting GT obj for {instance_name}.txt: existing: {existing_obj}, current_max_obj: {current_max_obj}')
        pickle.dump({'max_obj': current_max_obj * obj_multiplier, 'is_optimal': is_optimal}, f)
        return current_max_obj, is_optimal

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

def parse_log(instance_dict, log_file_path, si, suffix, obj_multiplier = 1.0, overwrite_min_lb = False):
    assert(obj_multiplier == 1.0 or obj_multiplier == -1.0)
    max_time = 0
    acc_sum_log = None
    count = 0
    log_data = open(log_file_path, "r").readlines()
    instance_to_locs = match_instance_name_to_lines(log_data, instance_dict['instances'], instance_dict['prefix'], suffix)
    if instance_to_locs is None:
        print(f'Unable to parse {log_file_path}')
        return
    print(f'Parsing {log_file_path}')

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
        logs_instance = np.concatenate((resampled, resampled_rel_gaps), 0) # time, clipped dual obj, clipped relative dual gaps.
        if acc_sum_log is not None:
            acc_sum_log = merge_logs(acc_sum_log, logs_instance, si)
        else:
            acc_sum_log = logs_instance
        count = count + 1
    acc_sum_log[1, :] /= count
    acc_sum_log[2, :] /= count 
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
        writer.writerow([valid_acc_avg_log[0, i], valid_acc_avg_log[1, i], valid_acc_avg_log[2, i]])
    f.close()

def parse_dataset(instance_dict, root_dir_or_log, suffix, si = 5, overwrite_min_lb = False, obj_multiplier = 1.0):
    if os.path.isfile(root_dir_or_log):
        parse_log(instance_dict, root_dir_or_log, si, suffix, obj_multiplier, overwrite_min_lb)
    else:
        for path, subdirs, files in os.walk(root_dir_or_log): # parse all log file in the directory.
            for filename in sorted(files):
                if not filename.endswith('.out'):
                    continue
                log_file = os.path.join(path, filename)
                parse_log(instance_dict, log_file, si, suffix, obj_multiplier, overwrite_min_lb)

#TODO Deal with our logs for instances where lowerbounds are converged but time is still increasing?
# parse_dataset(ct_instances, 
#             '/home/ahabbas/projects/LearnDBCA/out_dual/slurm_new/9282678.out', 
#             suffix_non_learned, si = 5, overwrite_min_lb = True)

# parse_dataset(ct_instances, 
#             '/home/ahabbas/projects/LearnDBCA/out_dual/slurm_new/9282678.out', 
#             suffix_learned, si = 5, overwrite_min_lb = False)

# parse_dataset(mrf_pf_instances, 
# '/home/ahabbas/projects/LearnDBCA/out_dual/MRF_PF/nobackup/v_new/v1_1_1_16_16_8_1_5_5_50_True_True_5e-3_False_2_True_True_0.1_False/eval_5_50_1e-9/9226107.out',
# suffix_non_learned, si = 10, overwrite_min_lb = True)
# parse_dataset(mrf_pf_instances, 
# '/home/ahabbas/projects/LearnDBCA/out_dual/MRF_PF/nobackup/v_new/v3_1_1_16_16_8_1_1_1_100_True_True_5e-3_False_2_True_True_0.1_False/eval_5_50_1e-9/9261837.out',
# suffix_learned, si = 10, overwrite_min_lb = False)

# parse_dataset(worms_instances, '/home/ahabbas/projects/LearnDBCA/out_dual/slurm/9282554_eval.out', suffix_non_learned, si = 2.5, overwrite_min_lb = True)
# parse_dataset(worms_instances, '/home/ahabbas/projects/LearnDBCA/out_dual/slurm/9282555_eval.out', suffix_learned, si = 2.5, overwrite_min_lb = False)

# parse_dataset(qaplib_small_instances, 
#             '/home/ahabbas/projects/LearnDBCA/out_dual/QAPLIB/nobackup/v_new2/v3_mixed_prec_lstm_gpu22_bs4_1_1_16_16_8_3_5_5_500_True_True_5e-4_False_2_True_True_True_10_0.0_True/double_prec_50_2000_0.0/9238987.out',
#             suffix_learned,
#             si = 5, 
#             overwrite_min_lb = True)

parse_dataset(ind_set_instances, 
            '/home/ahabbas/projects/LearnDBCA/out_dual/MIS/nobackup/vf/v6_mixed_prec_1_1_16_16_8_1_20_20_20_True_True_1e-3_False_1_True_True_0.0/double_prec_50_200_1e-9/9287189.out', 
            suffix_non_learned,
            si = 0.1, 
            overwrite_min_lb = True, 
            obj_multiplier = -1.0)

parse_dataset(ind_set_instances, 
            '/home/ahabbas/projects/LearnDBCA/out_dual/MIS/nobackup/vf/v6_mixed_prec_1_1_16_16_8_1_20_20_20_True_True_1e-3_False_1_True_True_0.0/double_prec_50_200_1e-9/9287189.out', 
            suffix_learned,
            si = 0.1, 
            overwrite_min_lb = False,
            obj_multiplier = -1.0)
