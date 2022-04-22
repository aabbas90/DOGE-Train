from yacs.config import CfgNode as CN
from config_dual.defaults import get_cfg_defaults, get_all_lp_instances

cfg = get_cfg_defaults()

cfg.LOG_EVERY = 100

cfg.TRAIN.MAX_NUM_EPOCHS = 50
cfg.DATA.NUM_WORKERS = 4
cfg.DATA.DATASETS = ['MIS_TRAIN', 'MIS_VAL']
cfg.DATA.VAL_FRACTION = [0.0, 1.0]
cfg.DATA.MIS_TRAIN_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/independent_set_random/train_split/', 'read_dual_converged': False, 'need_ilp_gt': False}) 
cfg.DATA.MIS_VAL_PARAMS = CN({'files_to_load': ['0.lp', '1.lp'], 'root_dir': '/home/ahabbas/data/learnDBCA/independent_set_random/train_split/', 'read_dual_converged' : False, 'need_ilp_gt': False}) 

cfg.TRAIN.BATCH_SIZE = 8
cfg.TRAIN.NUM_ROUNDS = 30
cfg.TRAIN.BASE_LR = 1e-3
cfg.TRAIN.USE_RELATIVE_GAP_LOSS = False
cfg.TRAIN.START_EPISODIC_TRAINING_AFTER_EPOCH = 5

cfg.TEST.NUM_ROUNDS = 20
cfg.TEST.NUM_DUAL_ITERATIONS = 50
cfg.TEST.BATCH_SIZE = 1
cfg.TEST.VAL_PERIOD = 10

test_datasets, test_params = get_all_lp_instances('/home/ahabbas/data/learnDBCA/independent_set_random/test_split/', 'MIS', None, False, True)
cfg.TEST.DATA.DATASETS = test_datasets
cfg.TEST.DATA.update(test_params)

cfg.OUTPUT_ROOT_DIR = '/home/ahabbas/projects/LearnDBCA/out_dual/' # Do not change, if changed exclude it from sbatch files from copying.
cfg.OUT_REL_DIR = 'MIS/v5/'