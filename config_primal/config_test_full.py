from yacs.config import CfgNode as CN
from config_primal.defaults import get_cfg_defaults

cfg = get_cfg_defaults()

cfg.DATA.DATASETS = ['worm07'] #'flywing_100_1', 
cfg.DATA.TEST_FRACTION = [1.0] # 
cfg.DATA.flywing_100_1_PARAMS = CN({'files_to_load': ['flywing_100_1.lp'], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/full_inst/', 'read_dual_converged' : True}) 
cfg.DATA.worm07_PARAMS = CN({'files_to_load': ['worm07-16-03-11-1745.lp'], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/full_inst/', 'read_dual_converged' : True}) 
cfg.MODEL.CKPT_PATH = 'out_primal/WORMS_SUBSET/v1_1_1_32_64_16_10_10_True_1e-3/default/version_0/checkpoints/last.ckpt'
cfg.DATA.NUM_WORKERS = 0

cfg.TEST.NUM_DUAL_ITERATIONS = 300
cfg.TEST.NUM_ROUNDS = 100
cfg.TEST.BATCH_SIZE = 1
