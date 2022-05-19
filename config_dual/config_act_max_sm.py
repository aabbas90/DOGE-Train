from yacs.config import CfgNode as CN
from config_dual.defaults import get_cfg_defaults, get_all_lp_instances

cfg = get_cfg_defaults()

cfg.LOG_EVERY = 100

cfg.MODEL.FULL_COORDINATE_ASCENT = False
cfg.MODEL.PREDICT_OMEGA = True
cfg.MODEL.PREDICT_DIST_WEIGHTS = True
cfg.MODEL.VAR_FEATURE_DIM = 16
cfg.MODEL.CON_FEATURE_DIM = 16
cfg.MODEL.EDGE_FEATURE_DIM = 8
cfg.MODEL.FEATURE_EXTRACTOR_DEPTH = 1
cfg.MODEL.DUAL_PRED_DEPTH = 1
cfg.MODEL.USE_LAYER_NORM = True
cfg.MODEL.NUM_HIDDEN_LAYERS_EDGE = 2
cfg.MODEL.USE_NET_SOLVER_COSTS = False
cfg.MODEL.USE_LSTM_VAR = False
cfg.MODEL.FREE_UPDATE = True
cfg.MODEL.DENORM_FREE_UPDATE = False
cfg.MODEL.SCALE_FREE_UPDATE = False
cfg.MODEL.USE_CELU_ACTIVATION = False
cfg.MODEL.USE_SEPARATE_MODEL_LATER_STAGE = False
cfg.MODEL.MP_AGGR = 'mean'

cfg.TRAIN.MAX_NUM_EPOCHS = 500
cfg.TRAIN.OPTIMIZER = "Adam"
cfg.DATA.NUM_WORKERS = 0
cfg.DATA.DATASETS = ['SM_TRAIN', 'SM_VAL']
cfg.DATA.VAL_FRACTION = [0.0, 1.0]
cfg.DATA.SM_TRAIN_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/shape_matching/train_split/', 'read_dual_converged' : False, 'need_gt': False, 'need_ilp_gt': False, 'load_in_memory': False}) 
cfg.DATA.SM_VAL_PARAMS = CN({'files_to_load': ['000003147572_861144_michael13_partial2_9992_michael15_partial5_17452_partial.lp'], 'root_dir': '/home/ahabbas/data/learnDBCA/shape_matching/test_split/', 'read_dual_converged' : False, 'need_gt': False, 'need_ilp_gt': False}) 
cfg.MODEL.MP_AGGR = 'max'

cfg.TRAIN.BATCH_SIZE = 2
cfg.TRAIN.NUM_ROUNDS = 30
cfg.TRAIN.BASE_LR = 1e-3
cfg.TRAIN.USE_RELATIVE_GAP_LOSS = False

cfg.TEST.NUM_ROUNDS = 20 # How many times rounding iterations.
cfg.TEST.NUM_DUAL_ITERATIONS = 100
cfg.TEST.BATCH_SIZE = 1

cfg.TEST.DATA.DATASETS = ['SM_TEST']
cfg.TEST.DATA.SM_TEST_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/shape_matching/train_split/', 'read_dual_converged' : False, 'need_gt': False, 'need_ilp_gt': False, 'load_in_memory': True})

cfg.OUTPUT_ROOT_DIR = '/home/ahabbas/projects/LearnDBCA/out_dual/'
cfg.OUT_REL_DIR = 'SM/v1/'