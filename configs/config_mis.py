from yacs.config import CfgNode as CN
from configs.defaults import get_cfg_defaults, get_all_lp_instances

cfg = get_cfg_defaults()

cfg.TRAIN.NUM_ROUNDS_WITH_GRAD = 1
cfg.TRAIN.NUM_DUAL_ITERATIONS = 20
cfg.TRAIN.GRAD_DUAL_ITR_MAX_ITR = 20
cfg.TRAIN.NUM_ROUNDS = 20
cfg.TRAIN.FREE_UPDATE_LOSS_WEIGHT = 0.0
cfg.TRAIN.MAX_NUM_EPOCHS = 300
cfg.TRAIN.NUM_JOURNEYS = 3
cfg.TRAIN.BATCH_SIZE = 8

cfg.MODEL.PREDICT_OMEGA = True
cfg.MODEL.PREDICT_DIST_WEIGHTS = True
cfg.MODEL.USE_LSTM_VAR = False
cfg.MODEL.FREE_UPDATE = True

cfg.TEST.DUAL_IMPROVEMENT_SLOPE = 0.0
cfg.TEST.NUM_ROUNDS = 500
cfg.TEST.NUM_DUAL_ITERATIONS = 100
cfg.TEST.BATCH_SIZE = 1
cfg.TEST.NUM_ROUNDS = 20
cfg.TEST.NUM_DUAL_ITERATIONS = 50
cfg.TEST.BATCH_SIZE = 1
cfg.TEST.VAL_PERIOD = 1000000

cfg.MODEL.CON_LP_FEATURES = ['lb', 'rhs', 'con_type', 'deg', 'prev_lb']
cfg.MODEL.CON_LP_FEATURES_INIT = ['lb', 'rhs', 'con_type', 'deg', 'prev_lb']
cfg.MODEL.EDGE_LP_FEATURES = ['sol', 'prev_sol', 'coeff', 'prev_sol_avg', 'mm_diff'] 
cfg.MODEL.EDGE_LP_FEATURES_INIT = ['sol', 'prev_sol', 'coeff', 'prev_sol_avg', 'mm_diff']

cfg.DATA.NUM_WORKERS = 4
cfg.DATA.DATASETS = ['MIS_TRAIN', 'MIS_VAL']
cfg.DATA.VAL_FRACTION = [0.0, 1.0]
cfg.DATA.MIS_TRAIN_PARAMS = CN({'files_to_load': [], 'root_dir': 'datasets/MIS/train_split/', 'read_dual_converged': False}) 
cfg.DATA.MIS_VAL_PARAMS = CN({'files_to_load': ['0.lp', '1.lp'], 'root_dir': 'datasets/MIS/train_split/', 'read_dual_converged' : False}) 

test_datasets, test_params = get_all_lp_instances(
    root_dir = 'datasets/MIS/test_split/', data_name = 'MIS', keyword = None, read_converged = False, need_gt = True)
cfg.TEST.DATA.DATASETS = test_datasets
cfg.TEST.DATA.update(test_params)

cfg.OUT_REL_DIR = 'MIS/v1/'