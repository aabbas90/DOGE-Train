from yacs.config import CfgNode as CN
from config_dual.defaults import get_cfg_defaults, get_all_lp_instances

cfg = get_cfg_defaults()

cfg.MODEL.FULL_COORDINATE_ASCENT = True
cfg.MODEL.VAR_LP_FEATURES = ['obj', 'deg']
cfg.MODEL.VAR_LP_FEATURES_INIT = ['obj', 'deg']
cfg.MODEL.CON_LP_FEATURES = ['lb', 'rhs', 'con_type', 'deg']
cfg.MODEL.CON_LP_FEATURES_INIT = ['lb', 'rhs', 'con_type', 'deg']
cfg.MODEL.EDGE_LP_FEATURES = ['sol', 'coeff', 'current_mm_diff']
cfg.MODEL.EDGE_LP_FEATURES_INIT = ['sol', 'coeff', 'mm_diff']

cfg.TRAIN.BATCH_SIZE = 12
cfg.TRAIN.NUM_ROUNDS = 10
cfg.TRAIN.NUM_DUAL_ITERATIONS = 50
cfg.TRAIN.START_EPISODIC_TRAINING_AFTER_EPOCH = 50
cfg.TRAIN.BASE_LR = 5e-4
cfg.TRAIN.MAX_NUM_EPOCHS = 150
cfg.DATA.NUM_WORKERS = 4
cfg.DATA.DATASETS = ['GM_WORMS_TRAIN', 'GM_WORMS_VAL']
cfg.DATA.VAL_FRACTION = [0.0, 1.0]
cfg.DATA.GM_WORMS_TRAIN_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/graph-matching/worms/train_split/', 'read_dual_converged' : True}) 
cfg.DATA.GM_WORMS_VAL_PARAMS = CN({'files_to_load': ['1.lp', '2.lp'], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/graph-matching/worms/test_split/', 'read_dual_converged' : True}) 

cfg.TEST.NUM_ROUNDS = 100
cfg.TEST.NUM_DUAL_ITERATIONS = 100
cfg.TEST.BATCH_SIZE = 1
cfg.TEST.VAL_PERIOD = 20 # Validate after every n epoch (can be less than 1).

test_datasets, test_params = get_all_lp_instances('/home/ahabbas/data/learnDBCA/cv_structure_pred/full_inst/', 'worms', 'worm', False)
cfg.TEST.DATA.DATASETS = test_datasets
cfg.TEST.DATA.update(test_params)

cfg.OUTPUT_ROOT_DIR = '/home/ahabbas/projects/LearnDBCA/out_dual/' # Do not change, if changed exclude it from sbatch files from copying.
cfg.OUT_REL_DIR = 'WORMS_FULL_COORD/v1_overfit/'
cfg.LOG_EVERY = 1