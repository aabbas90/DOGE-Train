from yacs.config import CfgNode as CN
from config_dual.defaults import get_cfg_defaults

cfg = get_cfg_defaults()

cfg.LOG_EVERY = 100

cfg.TRAIN.MAX_NUM_EPOCHS = 150
cfg.DATA.NUM_WORKERS = 4
# cfg.DATA.DATASETS = ['CT_SMALL', 'CT_LARGE'] #'GM_WORMS_TRAIN', 'GM_WORMS_TEST'] #
# cfg.DATA.TEST_FRACTION = [1.0, 0.0] #, 0.0, 1.0]
cfg.DATA.DATASETS = ['GM_WORMS_TRAIN', 'GM_WORMS_TEST'] #
cfg.DATA.TEST_FRACTION = [0.0, 1.0]
cfg.DATA.CT_SMALL_PARAMS = CN({'files_to_load': ['1.lp', '2.lp'], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/cell-tracking-AISTATS-2020/small/', 'read_dual_converged' : True}) 
cfg.DATA.CT_LARGE_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/cell-tracking-AISTATS-2020/large', 'read_dual_converged' : True}) 
cfg.DATA.GM_WORMS_TRAIN_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/graph-matching/worms/train_split/', 'read_dual_converged' : True}) 
cfg.DATA.GM_WORMS_TEST_PARAMS = CN({'files_to_load': ['1.lp', '2.lp'], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/graph-matching/worms/test_split/', 'read_dual_converged' : True}) 

cfg.TRAIN.NUM_ROUNDS = 30

cfg.TEST.NUM_ROUNDS = 10
cfg.TEST.PERIOD = 20
cfg.TEST.NUM_DUAL_ITERATIONS = 10
cfg.TEST.BATCH_SIZE = 1
cfg.TEST.PERIOD = 50 # Validate after every n epoch (can be less than 1).

cfg.OUTPUT_ROOT_DIR = '/home/ahabbas/projects/LearnDBCA/out_dual/' # Do not change, if changed exclude it from sbatch files from copying.
cfg.OUT_REL_DIR = 'WORMS/v1_conv/'