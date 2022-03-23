from yacs.config import CfgNode as CN
from config_primal.defaults import get_cfg_defaults, get_all_lp_instances

cfg = get_cfg_defaults()

cfg.LOG_EVERY = 1
cfg.DATA.DATASETS = []
cfg.DATA.VAL_FRACTION = []

cfg.TEST.DATA.DATASETS = ['GM_WORMS_CROPS']
cfg.TEST.DATA.GM_WORMS_CROPS_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/graph-matching/worms/', 'read_dual_converged' : True}) 

cfg.TEST.NUM_ROUNDS = 20 # How many times rounding iterations.
cfg.TEST.NUM_DUAL_ITERATIONS = 100
cfg.TEST.BATCH_SIZE = 1