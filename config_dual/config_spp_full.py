from yacs.config import CfgNode as CN
from config_dual.defaults import get_cfg_defaults, get_all_lp_instances

cfg = get_cfg_defaults()

cfg.LOG_EVERY = 100

cfg.TRAIN.MAX_NUM_EPOCHS = 300
cfg.DATA.NUM_WORKERS = 0
cfg.DATA.DATASETS = ['SPP_TRAIN', 'SPP_VAL']
cfg.DATA.VAL_FRACTION = [0.0, 1.0]

cfg.DATA.SPP_TRAIN_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/SPP_OR_Lib/train_split/', 'read_dual_converged' : False, 'need_gt': False, 'load_in_memory': True})
cfg.DATA.SPP_VAL_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/SPP_OR_Lib/test_split/', 'read_dual_converged' : False, 'need_gt': True})

cfg.TRAIN.BATCH_SIZE = 4
cfg.TRAIN.NUM_ROUNDS = 30
cfg.TRAIN.BASE_LR = 1e-3
cfg.TRAIN.USE_RELATIVE_GAP_LOSS = False

cfg.TEST.NUM_ROUNDS = 100
cfg.TEST.NUM_DUAL_ITERATIONS = 100
cfg.TEST.BATCH_SIZE = 1
cfg.TEST.VAL_PERIOD = 50 # Validate after every n epoch (can be less than 1). TODO

cfg.TEST.DATA.DATASETS = ['SPP_TEST']
cfg.TEST.DATA.SPP_TEST_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/SPP_OR_Lib/test_split/', 'read_dual_converged' : False, 'need_gt': True})

cfg.OUTPUT_ROOT_DIR = '/home/ahabbas/projects/LearnDBCA/out_dual/'
cfg.OUT_REL_DIR = 'SPP_Lib/v1/'