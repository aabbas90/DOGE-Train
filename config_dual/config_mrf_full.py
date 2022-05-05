from yacs.config import CfgNode as CN
from config_dual.defaults import get_cfg_defaults, get_all_lp_instances

cfg = get_cfg_defaults()

cfg.LOG_EVERY = 100

cfg.MODEL.CON_LP_FEATURES = ['lb', 'rhs', 'con_type', 'deg', 'prev_lb']
cfg.MODEL.CON_LP_FEATURES_INIT = ['lb', 'rhs', 'con_type', 'deg', 'prev_lb']
cfg.MODEL.EDGE_LP_FEATURES = ['sol', 'prev_sol', 'coeff', 'prev_sol_avg', 'mm_diff'] 
cfg.MODEL.EDGE_LP_FEATURES_INIT = ['sol', 'prev_sol', 'coeff', 'prev_sol_avg', 'mm_diff']

cfg.TRAIN.MAX_NUM_EPOCHS = 300
cfg.DATA.NUM_WORKERS = 4
cfg.DATA.DATASETS = ['MRF_TRAIN', 'MRF_VAL']
cfg.DATA.VAL_FRACTION = [0.0, 1.0]
MRF_TRAIN = ['objseg-349.lp', 'objseg-358.lp', 'objseg-416.lp']
cfg.DATA.MRF_TRAIN_PARAMS = CN({'files_to_load': MRF_TRAIN, 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/full_inst_mrf/', 'read_dual_converged' : False}) 
cfg.DATA.MRF_VAL_PARAMS = CN({'files_to_load': ['fourcolors.lp'], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/full_inst_mrf/', 'read_dual_converged' : False}) 

cfg.TRAIN.BATCH_SIZE = 2
cfg.TRAIN.NUM_ROUNDS = 30
cfg.TRAIN.BASE_LR = 1e-3
cfg.TRAIN.USE_RELATIVE_GAP_LOSS = False
cfg.TRAIN.START_EPISODIC_TRAINING_AFTER_EPOCH = 10

cfg.TEST.NUM_ROUNDS = 1000
cfg.TEST.NUM_DUAL_ITERATIONS = 5
cfg.TEST.BATCH_SIZE = 1
cfg.TEST.VAL_PERIOD = 100000

test_datasets, test_params = get_all_lp_instances('/home/ahabbas/data/learnDBCA/cv_structure_pred/full_inst_mrf/', 'MRF', 'objseg', False)
cfg.TEST.DATA.DATASETS = test_datasets
cfg.TEST.DATA.update(test_params)

cfg.OUTPUT_ROOT_DIR = '/home/ahabbas/projects/LearnDBCA/out_dual/' # Do not change, if changed exclude it from sbatch files from copying.
cfg.OUT_REL_DIR = 'MRF/v5/'