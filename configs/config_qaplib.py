from yacs.config import CfgNode as CN
from configs.defaults import get_cfg_defaults, get_all_lp_instances

cfg = get_cfg_defaults()

DATA_DIR = 'datasets/qaplib_full/small/' # TODO: Please set accordingly.

cfg.TRAIN.NUM_ROUNDS_WITH_GRAD = 1
cfg.TRAIN.NUM_DUAL_ITERATIONS = 5
cfg.TRAIN.GRAD_DUAL_ITR_MAX_ITR = 5
cfg.TRAIN.NUM_ROUNDS = 500
cfg.TRAIN.FREE_UPDATE_LOSS_WEIGHT = 0.0
cfg.TRAIN.BASE_LR = 1e-3
cfg.TRAIN.MAX_NUM_EPOCHS = 400
cfg.TRAIN.NUM_JOURNEYS = 4
cfg.TRAIN.BATCH_SIZE = 4

cfg.MODEL.PREDICT_OMEGA = True
cfg.MODEL.PREDICT_DIST_WEIGHTS = True
cfg.MODEL.USE_LSTM_VAR = False
cfg.MODEL.FREE_UPDATE = True

cfg.TEST.NUM_DUAL_ITERATIONS = 200
cfg.TEST.BATCH_SIZE = 1
cfg.TEST.VAL_PERIOD = 10000

cfg.MODEL.EDGE_LP_FEATURES = ['sol', 'prev_sol', 'coeff', 'mm_diff', 'prev_sol_avg']
cfg.MODEL.EDGE_LP_FEATURES_INIT = ['sol', 'prev_sol', 'coeff' , 'mm_diff', 'prev_sol_avg']

cfg.MODEL.CON_LP_FEATURES = ['lb', 'rhs', 'con_type', 'deg', 'prev_lb', 'lb_first_order_avg', 'lb_sec_order_avg', 'lb_change_free_update']
cfg.MODEL.CON_LP_FEATURES_INIT = ['lb', 'rhs', 'con_type', 'deg', 'prev_lb', 'lb_first_order_avg', 'lb_sec_order_avg', 'lb_change']

cfg.TRAIN.MAX_NUM_EPOCHS = 1000
cfg.DATA.NUM_WORKERS = 0
cfg.DATA.DATASETS = ['QAP_TRAIN', 'QAP_VAL']
cfg.DATA.VAL_FRACTION = [0.0, 1.0]

cfg.DATA.QAP_TRAIN_PARAMS = CN({'files_to_load': [], 'root_dir': 'f{DATA_DIR}/train_split', 'read_dual_converged' : False, 'need_gt': False, 'load_in_memory': True})
cfg.DATA.QAP_VAL_PARAMS = CN({'files_to_load': ['chr22b.lp', 'had20.lp', 'nug30.lp'], 'root_dir': 'f{DATA_DIR}/test_split/', 'read_dual_converged' : False, 'need_gt': False})

cfg.TRAIN.BATCH_SIZE = 4
cfg.TRAIN.NUM_ROUNDS = 30
cfg.TRAIN.BASE_LR = 1e-3
cfg.TRAIN.USE_RELATIVE_GAP_LOSS = False

cfg.TEST.NUM_ROUNDS = 5000
cfg.TEST.NUM_DUAL_ITERATIONS = 20
cfg.TEST.BATCH_SIZE = 1
cfg.TEST.VAL_PERIOD = 5000 # Validate after every n epoch (can be less than 1). TODO

cfg.TEST.DATA.DATASETS = ['QAP_TEST']
cfg.TEST.DATA.QAP_TEST_PARAMS = CN({'files_to_load': [], 'root_dir': 'f{DATA_DIR}/test_split/', 'read_dual_converged' : False, 'need_gt': False})

cfg.OUTPUT_ROOT_DIR = 'output_logs/'
cfg.OUT_REL_DIR = 'QAPLIB/v1/'