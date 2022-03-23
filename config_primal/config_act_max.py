from yacs.config import CfgNode as CN
from config_primal.defaults import get_cfg_defaults, get_all_lp_instances

cfg = get_cfg_defaults()

cfg.LOG_EVERY = 1
cfg.DATA.DATASETS = []
cfg.DATA.VAL_FRACTION = []

test_datasets, test_params = get_all_lp_instances('/home/ahabbas/data/learnDBCA/cv_structure_pred/full_inst/', 'worms', 'worm')
cfg.TEST.DATA.DATASETS = test_datasets
cfg.TEST.DATA.update(test_params)

cfg.TEST.NUM_ROUNDS = 20 # How many times rounding iterations.
cfg.TEST.NUM_DUAL_ITERATIONS = 100
cfg.TEST.BATCH_SIZE = 1