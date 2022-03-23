from yacs.config import CfgNode as CN
from config_dual.defaults import get_cfg_defaults, get_all_lp_instances

cfg = get_cfg_defaults()

cfg.LOG_EVERY = 100

cfg.MODEL.VAR_LP_FEATURES = ['obj', 'deg']
cfg.MODEL.VAR_LP_FEATURES_INIT = ['obj', 'deg']
cfg.MODEL.CON_LP_FEATURES = ['lb', 'deg']
cfg.MODEL.CON_LP_FEATURES_INIT = ['lb', 'deg']
cfg.MODEL.EDGE_LP_FEATURES = ['sol', 'prev_dist_weights']
cfg.MODEL.EDGE_LP_FEATURES_INIT = ['sol', 'dist_weights']

cfg.TRAIN.MAX_NUM_EPOCHS = 100
cfg.DATA.NUM_WORKERS = 4
cfg.DATA.DATASETS = ['MIPLIB_TRAIN', 'MIPLIB_VAL']
cfg.DATA.VAL_FRACTION = [0.0, 1.0]
cfg.DATA.MIPLIB_TRAIN_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/miplib_crops/easy/', 'read_dual_converged' : True, 'need_gt': True, 'need_ilp_gt': False}) 
cfg.DATA.MIPLIB_VAL_PARAMS = CN({'files_to_load': ['1.lp', '2.lp', '3.lp'], 'root_dir': '/home/ahabbas/data/learnDBCA/miplib_crops/easy/', 'read_dual_converged' : True, 'need_gt': True, 'need_ilp_gt': False})

cfg.TRAIN.NUM_ROUNDS = 30

cfg.TEST.NUM_ROUNDS = 10
cfg.TEST.NUM_DUAL_ITERATIONS = 1000
cfg.TEST.BATCH_SIZE = 1
cfg.TEST.VAL_PERIOD = 10 # Validate after every n epoch (can be less than 1).

test_datasets, test_params = get_all_lp_instances('/home/ahabbas/data/learnDBCA/miplib/easy/', 'miplib_easy', None, False)
cfg.TEST.DATA.DATASETS = test_datasets
cfg.TEST.DATA.update(test_params)

cfg.OUTPUT_ROOT_DIR = '/home/ahabbas/projects/LearnDBCA/out_dual/' # Do not change, if changed exclude it from sbatch files from copying.
cfg.OUT_REL_DIR = 'MIPLIB/v1/'