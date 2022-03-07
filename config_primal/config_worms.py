from yacs.config import CfgNode as CN
from config_primal.defaults import get_cfg_defaults

cfg = get_cfg_defaults()

cfg.MODEL.PREDICT_DIST_WEIGHTS = False

cfg.LOG_EVERY = 10
cfg.DATA.DATASETS = ['GM_WORMS_TRAIN', 'GM_WORMS_TEST']
cfg.DATA.TEST_FRACTION = [0.0, 1.0]

cfg.DATA.GM_WORMS_TRAIN_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/graph-matching/worms/train_split/', 'read_dual_converged' : True}) 
cfg.DATA.GM_WORMS_TEST_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/graph-matching/worms/test_split/', 'read_dual_converged' : True}) 

cfg.TRAIN.BATCH_SIZE = 8
cfg.TRAIN.BASE_LR = 1e-4
cfg.TRAIN.MAX_NUM_EPOCHS = 100
cfg.TRAIN.START_EPISODIC_TRAINING_AFTER_EPOCH = 20
cfg.TRAIN.NUM_ROUNDS = 20 # Max. possible number of rounding iterations.
cfg.TRAIN.LOSS_MARGIN = 5e-3

cfg.TEST.NUM_DUAL_ITERATIONS = 200
cfg.TEST.NUM_ROUNDS = 20 # How many times rounding iterations.
cfg.TEST.BATCH_SIZE = 1
cfg.TEST.PERIOD = 50 # Validate after every n epoch (can be less than 1).
