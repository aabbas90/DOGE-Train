from yacs.config import CfgNode as CN
from config_primal.defaults import get_cfg_defaults
import os
cfg = get_cfg_defaults()

def get_all_lp_instances(root_dir):
    datasets = []
    all_params = {}
    for path, subdirs, files in os.walk(root_dir):
        for instance_name in sorted(files):
            if not instance_name.endswith('.lp') or 'nan' in instance_name or 'normalized' in instance_name:
                continue

            data_name = instance_name.replace('.lp', '')
            datasets.append(data_name)
            all_params[data_name + '_PARAMS'] = CN({
                    'root_dir': root_dir, 
                    'files_to_load': instance_name,
                    'read_dual_converged' : True,
                    'need_gt': False}) 
    return datasets, all_params

root_dir = '/home/ahabbas/data/learnDBCA/cv_structure_pred/full_inst/'
datasets, all_params = get_all_lp_instances(root_dir)

cfg.DATA.DATASETS = datasets
cfg.DATA.TEST_FRACTION = [1.0] * len(datasets)
cfg.DATA.update(all_params)

cfg.MODEL.CKPT_PATH = 'out_primal/CT/v1_sep_metrics_1_1_8_16_8_3/default/version_1/checkpoints/last.ckpt'
cfg.TEST.NUM_DUAL_ITERATIONS = 500
cfg.TEST.NUM_ROUNDS = 50
cfg.TEST.BATCH_SIZE = 1
