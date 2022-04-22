from yacs.config import CfgNode as CN
from config_primal.defaults import get_cfg_defaults, get_all_lp_instances

cfg = get_cfg_defaults()

cfg.LOG_EVERY = 1
cfg.DATA.DATASETS = ['GM_WORMS_TRAIN', 'GM_WORMS_VAL']
cfg.DATA.VAL_FRACTION = [0.0, 1.0]
WORM_TRAIN = [f'worm0{i}-16-03-11-1745.lp' for i in range(10)]
cfg.DATA.GM_WORMS_TRAIN_PARAMS = CN({'files_to_load': WORM_TRAIN, 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/full_inst/', 'read_dual_converged' : True}) 
cfg.DATA.GM_WORMS_VAL_PARAMS = CN({'files_to_load': ['1.lp'], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/graph-matching/worms/test_split/', 'read_dual_converged' : True}) 

test_datasets, test_params = get_all_lp_instances('/home/ahabbas/data/learnDBCA/cv_structure_pred/full_inst/', 'worms', 'worm')
cfg.TEST.DATA.DATASETS = test_datasets
cfg.TEST.DATA.update(test_params)

cfg.TRAIN.BATCH_SIZE = 1
cfg.TRAIN.LOSS_DISCOUNT_FACTOR = 0.9
cfg.TRAIN.BASE_LR = 1e-4
cfg.TRAIN.MAX_NUM_EPOCHS = 100
cfg.TRAIN.START_EPISODIC_TRAINING_AFTER_EPOCH = 20
cfg.TRAIN.NUM_ROUNDS = 5 # Max. possible number of rounding iterations.
cfg.TRAIN.DUAL_IMPROVEMENT_SLOPE = 1e-6
cfg.TRAIN.MIN_PERTURBATION = 1e-2

cfg.TEST.NUM_ROUNDS = 20 # How many times rounding iterations.
cfg.TEST.NUM_DUAL_ITERATIONS = 500
cfg.TEST.BATCH_SIZE = 1
cfg.TEST.VAL_PERIOD = 50 # Validate after every n epoch (can be less than 1).