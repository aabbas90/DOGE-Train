from yacs.config import CfgNode as CN
from config_dual.defaults import get_cfg_defaults, get_all_lp_instances

cfg = get_cfg_defaults()

cfg.LOG_EVERY = 100

cfg.MODEL.EDGE_LP_FEATURES = ['sol', 'prev_sol', 'coeff', 'mm_diff', 'prev_sol_avg']
cfg.MODEL.EDGE_LP_FEATURES_INIT = ['sol', 'prev_sol', 'coeff' , 'mm_diff', 'prev_sol_avg']

cfg.MODEL.CON_LP_FEATURES = ['lb', 'rhs', 'con_type', 'deg', 'prev_lb', 'lb_first_order_avg', 'lb_sec_order_avg', 'lb_change_free_update']
cfg.MODEL.CON_LP_FEATURES_INIT = ['lb', 'rhs', 'con_type', 'deg', 'prev_lb', 'lb_first_order_avg', 'lb_sec_order_avg', 'lb_change']

cfg.TRAIN.MAX_NUM_EPOCHS = 500
cfg.DATA.NUM_WORKERS = 0
cfg.DATA.DATASETS = ['UNI_TRAIN', 'UNI_VAL']
cfg.DATA.VAL_FRACTION = [0.0, 1.0]

cfg.DATA.UNI_TRAIN_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/universal_model/train_split', 'read_dual_converged' : False, 'need_gt': False, 'load_in_memory': True})
cfg.DATA.UNI_VAL_PARAMS = CN({'files_to_load': ['chr22b.lp', 'had20.lp'], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/qaplib_full/small/test_split/', 'read_dual_converged' : False, 'need_gt': False})

cfg.TRAIN.BATCH_SIZE = 4
cfg.TRAIN.NUM_ROUNDS = 30
cfg.TRAIN.BASE_LR = 1e-3
cfg.TRAIN.USE_RELATIVE_GAP_LOSS = False

cfg.TEST.NUM_ROUNDS = 2000
cfg.TEST.NUM_DUAL_ITERATIONS = 50
cfg.TEST.BATCH_SIZE = 1
cfg.TEST.VAL_PERIOD = 50000 # Validate after every n epoch (can be less than 1).

cfg.TEST.DATA.DATASETS = ['CT_TEST', 'QAP_TEST', 'MIS_TEST', 'WORM_TEST']
CT_TEST_FILES =['flywing_245.lp'] #, 'flywing_100_1.lp', 'flywing_100_2.lp']
cfg.TEST.DATA.CT_TEST_PARAMS = CN({'files_to_load': CT_TEST_FILES, 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/full_inst/', 'read_dual_converged' : False, 'need_gt': False})

cfg.TEST.DATA.QAP_TEST_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/qaplib_full/small/test_split/', 'read_dual_converged' : False, 'need_gt': False})
cfg.TEST.DATA.MIS_TEST_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/independent_set_random/test_split/', 'read_dual_converged' : False, 'need_gt': False})

WORM_TEST = [f'worm{i}-16-03-11-1745.lp' for i in range(10, 31)]
cfg.TEST.DATA.WORM_TEST_PARAMS = CN({'files_to_load': WORM_TEST, 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/full_inst/', 'read_dual_converged' : False}) 

cfg.OUTPUT_ROOT_DIR = '/home/ahabbas/projects/LearnDBCA/out_dual/'
cfg.OUT_REL_DIR = 'universal/v1/'