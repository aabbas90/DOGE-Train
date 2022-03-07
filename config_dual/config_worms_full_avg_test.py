from yacs.config import CfgNode as CN
from config_primal.defaults import get_cfg_defaults
import os
cfg = get_cfg_defaults()

def get_all_lp_instances(root_dir):
    data_name = 'GM_WORMS_AVG'
    datasets = [data_name]
    files_to_load = []
    for path, subdirs, files in os.walk(root_dir):
        for instance_name in sorted(files):
            if not instance_name.endswith('.lp') or 'nan' in instance_name or 'normalized' in instance_name:
                continue

            if not 'worm' in instance_name:
                continue

            files_to_load.append(instance_name)

    all_params = {}
    all_params[data_name + '_PARAMS'] = CN({
            'root_dir': root_dir, 
            'files_to_load': files_to_load,
            'read_dual_converged' : False,
            'need_gt': False})
    return datasets, all_params

root_dir = '/home/ahabbas/data/learnDBCA/cv_structure_pred/full_inst/'
datasets, all_params = get_all_lp_instances(root_dir)

cfg.DATA.DATASETS = datasets
cfg.DATA.TEST_FRACTION = [1.0] * len(datasets)
cfg.DATA.update(all_params)

cfg.TEST.BATCH_SIZE = 1
