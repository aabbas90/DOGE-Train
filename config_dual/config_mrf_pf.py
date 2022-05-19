from yacs.config import CfgNode as CN
from config_dual.defaults import get_cfg_defaults, get_all_lp_instances

cfg = get_cfg_defaults()

cfg.LOG_EVERY = 100

cfg.MODEL.EDGE_LP_FEATURES = ['sol', 'prev_sol', 'coeff', 'mm_diff']
cfg.MODEL.EDGE_LP_FEATURES_INIT = ['sol', 'prev_sol', 'coeff' , 'mm_diff']

cfg.MODEL.CON_LP_FEATURES = ['lb', 'rhs', 'con_type', 'deg', 'prev_lb', 'lb_first_order_avg', 'lb_sec_order_avg', 'lb_change_free_update']
cfg.MODEL.CON_LP_FEATURES_INIT = ['lb', 'rhs', 'con_type', 'deg', 'prev_lb', 'lb_first_order_avg', 'lb_sec_order_avg', 'lb_change']

cfg.MODEL.MP_AGGR = 'max'

cfg.TRAIN.MAX_NUM_EPOCHS = 100
cfg.DATA.NUM_WORKERS = 0
cfg.DATA.DATASETS = ['MRF_PF_TRAIN', 'MRF_PF_VAL']
cfg.DATA.VAL_FRACTION = [0.0, 1.0]
cfg.DATA.MRF_PF_TRAIN_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/mrf/protein_folding/train_split_large/', 'read_dual_converged' : False, 'need_gt': False, 'need_ilp_gt': False, 'load_in_memory': True}) 
cfg.DATA.MRF_PF_VAL_PARAMS = CN({'files_to_load': ['pdb1b25.lp'], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/mrf/protein_folding/test_split/', 'read_dual_converged' : False, 'need_gt': False, 'need_ilp_gt': False}) 

cfg.TRAIN.BATCH_SIZE = 1
cfg.TRAIN.NUM_ROUNDS = 30
cfg.TRAIN.BASE_LR = 1e-3
cfg.TRAIN.USE_RELATIVE_GAP_LOSS = False

cfg.TEST.NUM_ROUNDS = 100
cfg.TEST.NUM_DUAL_ITERATIONS = 50
cfg.TEST.BATCH_SIZE = 1
cfg.TEST.VAL_PERIOD = 10000 # Validate after every n epoch (can be less than 1).

cfg.TEST.DATA.DATASETS = ['MRF_PF_TEST']
cfg.TEST.DATA.MRF_PF_TEST_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/mrf/protein_folding/test_split_large/', 'read_dual_converged' : False, 'need_gt': False, 'need_ilp_gt': False})

cfg.OUTPUT_ROOT_DIR = '/home/ahabbas/projects/LearnDBCA/out_dual/'
cfg.OUT_REL_DIR = 'MRF_PF/v1/'