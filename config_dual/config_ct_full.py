from yacs.config import CfgNode as CN
from config_dual.defaults import get_cfg_defaults, get_all_lp_instances

cfg = get_cfg_defaults()

cfg.LOG_EVERY = 100

cfg.TRAIN.MAX_NUM_EPOCHS = 400
cfg.DATA.NUM_WORKERS = 4
cfg.DATA.DATASETS = ['CT_TRAIN', 'CT_VAL']
cfg.DATA.VAL_FRACTION = [0.0, 1.0]

#CT_TRAIN_FILES = ['flywing_11', 'PhC-C2DH-U373_01.lp', 'PhC-C2DH-U373_02.lp', 'Fluo-N2DL-HELA.lp', 'drosophila.lp', 'Fluo-C2DL-MSC_01.lp']
CT_TRAIN_FILES = ['flywing_100_1.lp', 'flywing_100_2.lp']
cfg.DATA.CT_TRAIN_PARAMS = CN({'files_to_load': CT_TRAIN_FILES, 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/full_inst/', 'read_dual_converged' : False, 'need_gt': True})
cfg.DATA.CT_VAL_PARAMS = CN({'files_to_load': ['0.lp'], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/cell-tracking-AISTATS-2020/large/flywing_100_2/', 'read_dual_converged' : False}) 

cfg.TRAIN.BATCH_SIZE = 1
cfg.TRAIN.NUM_ROUNDS = 30
cfg.TRAIN.BASE_LR = 1e-3
cfg.TRAIN.USE_RELATIVE_GAP_LOSS = False

cfg.TEST.NUM_ROUNDS = 20
cfg.TEST.NUM_DUAL_ITERATIONS = 500
cfg.TEST.BATCH_SIZE = 1
cfg.TEST.VAL_PERIOD = 50 # Validate after every n epoch (can be less than 1). TODO

cfg.TEST.DATA.DATASETS = ['CT_TEST']
CT_TEST_FILES =['flywing_245.lp', 'flywing_100_1.lp', 'flywing_100_2.lp']
cfg.TEST.DATA.CT_TEST_PARAMS = CN({'files_to_load': CT_TEST_FILES, 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/full_inst/', 'read_dual_converged' : False, 'need_gt': False})

cfg.OUTPUT_ROOT_DIR = '/home/ahabbas/projects/LearnDBCA/out_dual/'
cfg.OUT_REL_DIR = 'CT_LARGE/v5/'