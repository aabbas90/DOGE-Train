from yacs.config import CfgNode as CN
from config_dual.defaults import get_cfg_defaults, get_all_lp_instances

cfg = get_cfg_defaults()

cfg.LOG_EVERY = 100

cfg.TRAIN.MAX_NUM_EPOCHS = 500
cfg.DATA.NUM_WORKERS = 4
cfg.DATA.DATASETS = ['QAP_TRAIN', 'QAP_VAL']
cfg.DATA.VAL_FRACTION = [0.0, 1.0]

cfg.DATA.QAP_TRAIN_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/qaplib_full/small/train_split', 'read_dual_converged' : False, 'need_gt': False})
cfg.DATA.QAP_VAL_PARAMS = CN({'files_to_load': ['chr22b.lp', 'had20.lp', 'nug30.lp'], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/qaplib_full/small/test_split/', 'read_dual_converged' : False, 'need_gt': False})

cfg.TRAIN.BATCH_SIZE = 8
cfg.TRAIN.NUM_ROUNDS = 30
cfg.TRAIN.BASE_LR = 1e-3
cfg.TRAIN.USE_RELATIVE_GAP_LOSS = False

cfg.TEST.NUM_ROUNDS = 20
cfg.TEST.NUM_DUAL_ITERATIONS = 500
cfg.TEST.BATCH_SIZE = 1
cfg.TEST.VAL_PERIOD = 5000 # Validate after every n epoch (can be less than 1). TODO

cfg.TEST.DATA.DATASETS = ['QAP_TEST']
cfg.TEST.DATA.QAP_TEST_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/qaplib_full/small/test_split/', 'read_dual_converged' : False, 'need_gt': False})

cfg.OUTPUT_ROOT_DIR = '/home/ahabbas/projects/LearnDBCA/out_dual/'
cfg.OUT_REL_DIR = 'QAPLIB/v1/'