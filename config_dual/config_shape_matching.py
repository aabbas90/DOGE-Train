from yacs.config import CfgNode as CN
from config_dual.defaults import get_cfg_defaults, get_all_lp_instances

cfg = get_cfg_defaults()

cfg.LOG_EVERY = 100

cfg.MODEL.EDGE_LP_FEATURES = ['sol', 'prev_sol', 'coeff', 'prev_sol_avg', 'mm_diff']
cfg.MODEL.EDGE_LP_FEATURES_INIT = ['sol', 'prev_sol', 'coeff' , 'prev_sol_avg', 'mm_diff']

cfg.MODEL.CON_LP_FEATURES = ['lb', 'rhs', 'con_type', 'deg', 'prev_lb', 'lb_first_order_avg', 'lb_sec_order_avg', 'lb_change_free_update']
cfg.MODEL.CON_LP_FEATURES_INIT = ['lb', 'rhs', 'con_type', 'deg', 'prev_lb', 'lb_first_order_avg', 'lb_sec_order_avg', 'lb_change']

cfg.TRAIN.MAX_NUM_EPOCHS = 200
cfg.TRAIN.OPTIMIZER = "Adam"
cfg.DATA.NUM_WORKERS = 0
cfg.DATA.DATASETS = ['SM_TRAIN', 'SM_VAL']
cfg.DATA.VAL_FRACTION = [0.0, 1.0]
cfg.DATA.SM_TRAIN_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/shape_matching/train_split/', 'read_dual_converged' : False, 'need_gt': False, 'need_ilp_gt': False, 'load_in_memory': True}) 
cfg.DATA.SM_VAL_PARAMS = CN({'files_to_load': ['000003147572_861144_michael13_partial2_9992_michael15_partial5_17452_partial.lp'], 'root_dir': '/home/ahabbas/data/learnDBCA/shape_matching/test_split/', 'read_dual_converged' : False, 'need_gt': False, 'need_ilp_gt': False}) 
cfg.MODEL.MP_AGGR = 'mean'

cfg.TRAIN.BATCH_SIZE = 1
cfg.TRAIN.NUM_ROUNDS = 30
cfg.TRAIN.BASE_LR = 1e-3
cfg.TRAIN.USE_RELATIVE_GAP_LOSS = False

cfg.TEST.NUM_ROUNDS = 25
cfg.TEST.NUM_DUAL_ITERATIONS = 500
cfg.TEST.BATCH_SIZE = 1
cfg.TEST.VAL_PERIOD = 10000
cfg.TEST.DUAL_IMPROVEMENT_SLOPE = 1e-9

cfg.TEST.DATA.DATASETS = ['SM_TEST']
cfg.TEST.DATA.SM_TEST_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/shape_matching/test_split/', 'read_dual_converged' : False, 'need_gt': False, 'need_ilp_gt': False})

cfg.OUTPUT_ROOT_DIR = '/home/ahabbas/projects/LearnDBCA/out_dual/'
cfg.OUT_REL_DIR = 'SM/v1/'