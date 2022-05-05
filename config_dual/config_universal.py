from yacs.config import CfgNode as CN
from config_dual.defaults import get_cfg_defaults, get_all_lp_instances

cfg = get_cfg_defaults()

cfg.LOG_EVERY = 100

cfg.TRAIN.MAX_NUM_EPOCHS = 200
cfg.DATA.NUM_WORKERS = 4
cfg.DATA.DATASETS = ['GM_WORMS_TRAIN', 'GM_WORMS_VAL', 'QAP_TRAIN', 'QAP_VAL', 'MIS_TRAIN', 'MIS_VAL', 'CT_TRAIN', 'CT_VAL']
cfg.DATA.VAL_FRACTION = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

cfg.DATA.GM_WORMS_TRAIN_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/graph-matching/worms/train_split/', 'read_dual_converged': False}) 
cfg.DATA.GM_WORMS_VAL_PARAMS = CN({'files_to_load': ['1.lp'], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/graph-matching/worms/test_split/', 'read_dual_converged' : False}) 
cfg.DATA.QAP_TRAIN_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/qaplib_full/small/train_split', 'read_dual_converged' : False, 'need_gt': False, 'load_in_memory': False})
cfg.DATA.QAP_VAL_PARAMS = CN({'files_to_load': ['chr22b.lp', 'had20.lp', 'nug30.lp'], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/qaplib_full/small/test_split/', 'read_dual_converged' : False, 'need_gt': False})
cfg.DATA.MIS_TRAIN_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/independent_set_random/train_split/', 'read_dual_converged': False, 'need_ilp_gt': False}) 
cfg.DATA.MIS_VAL_PARAMS = CN({'files_to_load': ['0.lp', '1.lp'], 'root_dir': '/home/ahabbas/data/learnDBCA/independent_set_random/train_split/', 'read_dual_converged' : False, 'need_ilp_gt': False}) 
CT_TRAIN_FILES = ['flywing_100_1.lp', 'flywing_100_2.lp']
cfg.DATA.CT_TRAIN_PARAMS = CN({'files_to_load': CT_TRAIN_FILES, 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/full_inst/', 'read_dual_converged' : False, 'need_gt': True, 'load_in_memory': False})
cfg.DATA.CT_VAL_PARAMS = CN({'files_to_load': ['0.lp'], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/cell-tracking-AISTATS-2020/large/flywing_100_2/', 'read_dual_converged' : False}) 

cfg.TRAIN.BATCH_SIZE = 4
cfg.TRAIN.NUM_ROUNDS = 30
cfg.TRAIN.BASE_LR = 1e-3
cfg.TRAIN.USE_RELATIVE_GAP_LOSS = False

cfg.TEST.NUM_ROUNDS = 20
cfg.TEST.NUM_DUAL_ITERATIONS = 500
cfg.TEST.BATCH_SIZE = 1
cfg.TEST.VAL_PERIOD = 1000

cfg.TEST.DATA.DATASETS = ['WORM_TEST', 'QAP_TEST', 'CT_TEST']
WORM_TEST = [f'worm{i}-16-03-11-1745.lp' for i in range(10, 31)]
cfg.TEST.DATA.WORM_TEST_PARAMS = CN({'files_to_load': WORM_TEST, 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/full_inst/', 'read_dual_converged' : False}) 

test_datasets, test_params = get_all_lp_instances('/home/ahabbas/data/learnDBCA/independent_set_random/test_split/', 'MIS', None, False, False)
cfg.TEST.DATA.DATASETS.append(test_datasets)
cfg.TEST.DATA.update(test_params)

cfg.TEST.DATA.QAP_TEST_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/qaplib_full/small/test_split/', 'read_dual_converged' : False, 'need_gt': False})

CT_TEST_FILES =['flywing_245.lp']
cfg.TEST.DATA.CT_TEST_PARAMS = CN({'files_to_load': CT_TEST_FILES, 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/full_inst/', 'read_dual_converged' : False, 'need_gt': False})

cfg.OUTPUT_ROOT_DIR = '/home/ahabbas/projects/LearnDBCA/out_dual/' # Do not change, if changed exclude it from sbatch files from copying.
cfg.OUT_REL_DIR = 'WORMS/v5/'