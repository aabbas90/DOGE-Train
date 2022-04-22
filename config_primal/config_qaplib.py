from yacs.config import CfgNode as CN
from config_primal.defaults import get_cfg_defaults, get_all_lp_instances

cfg = get_cfg_defaults()

cfg.LOG_EVERY = 1
cfg.DATA.DATASETS = ['QAP_TRAIN', 'QAP_VAL']
cfg.DATA.VAL_FRACTION = [0.0, 1.0]

cfg.DATA.QAP_TRAIN_PARAMS = CN({'files_to_load': ['ste36a.lp'], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/qaplib_full/small/train_split', 'read_dual_converged' : True})
cfg.DATA.QAP_VAL_PARAMS = CN({'files_to_load': ['chr22b.lp', 'had20.lp', 'nug30.lp'], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/qaplib_full/small/test_split/', 'read_dual_converged' : True})

cfg.TEST.DATA.DATASETS = ['QAP_TEST']
cfg.TEST.DATA.QAP_TEST_PARAMS = CN({'files_to_load': ['ste36a.lp'], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/qaplib_full/small/train_split/', 'read_dual_converged' : True})

cfg.TRAIN.BATCH_SIZE = 1
cfg.TRAIN.LOSS_DISCOUNT_FACTOR = 0.9
cfg.TRAIN.BASE_LR = 1e-4
cfg.TRAIN.MAX_NUM_EPOCHS = 1000
cfg.TRAIN.START_EPISODIC_TRAINING_AFTER_EPOCH = 20
cfg.TRAIN.NUM_ROUNDS = 5 # Max. possible number of rounding iterations.
cfg.TRAIN.DUAL_IMPROVEMENT_SLOPE = 1e-6
cfg.TRAIN.MIN_PERTURBATION = 1e-2

cfg.TEST.NUM_ROUNDS = 100 # How many times rounding iterations.
cfg.TEST.NUM_DUAL_ITERATIONS = 500
cfg.TEST.BATCH_SIZE = 1
cfg.TEST.VAL_PERIOD = 1000 # Validate after every n epoch (can be less than 1).