from yacs.config import CfgNode as CN
from config_primal.defaults import get_cfg_defaults, get_all_lp_instances

cfg = get_cfg_defaults()

cfg.MODEL.PREDICT_DIST_WEIGHTS = False

cfg.LOG_EVERY = 1
cfg.DATA.DATASETS = ['GM_WORMS_TRAIN', 'GM_WORMS_VAL']
cfg.DATA.VAL_FRACTION = [0.0, 1.0]
cfg.DATA.GM_WORMS_TRAIN_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/graph-matching/worms/train_split/', 'read_dual_converged' : True}) 
cfg.DATA.GM_WORMS_VAL_PARAMS = CN({'files_to_load': ['0.lp', '1.lp', '2.lp'], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/graph-matching/worms/test_split/', 'read_dual_converged' : True}) 

test_datasets, test_params = get_all_lp_instances('/home/ahabbas/data/learnDBCA/cv_structure_pred/full_inst/', 'worms', 'worm')
cfg.TEST.DATA.DATASETS = test_datasets
cfg.TEST.DATA.update(test_params)

cfg.TRAIN.BATCH_SIZE = 8
cfg.TRAIN.BASE_LR = 1e-4
cfg.TRAIN.MAX_NUM_EPOCHS = 1000
cfg.TRAIN.START_EPISODIC_TRAINING_AFTER_EPOCH = 20
cfg.TRAIN.NUM_ROUNDS = 5 # Max. possible number of rounding iterations.
cfg.TRAIN.LOSS_MARGIN = 5e-3
cfg.TRAIN.DUAL_IMPROVEMENT_SLOPE = 1e-6
cfg.TRAIN.MIN_PERTURBATION = 1e-2

cfg.TEST.NUM_ROUNDS = 10 # How many times rounding iterations.
cfg.TEST.NUM_DUAL_ITERATIONS = 200
cfg.TEST.BATCH_SIZE = 1
cfg.TEST.VAL_PERIOD = 50 # Validate after every n epoch (can be less than 1).