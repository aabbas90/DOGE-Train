from yacs.config import CfgNode as CN

cfg = CN()
cfg.DEVICE = 'gpu'

cfg.MODEL = CN()
cfg.MODEL.VAR_FEATURE_DIM = 16
cfg.MODEL.CON_FEATURE_DIM = 16
cfg.MODEL.EDGE_FEATURE_DIM = 8
cfg.MODEL.FEATURE_EXTRACTOR_DEPTH = 3
cfg.MODEL.PRIMAL_PRED_DEPTH = 1
cfg.MODEL.CKPT_PATH = None
cfg.MODEL.OMEGA = 0.5
cfg.MODEL.VAR_LP_FEATURES = ['obj', 'deg', 'net_pert']
cfg.MODEL.CON_LP_FEATURES = ['lb', 'rhs', 'con_type', 'deg']
cfg.MODEL.EDGE_LP_FEATURES = ['lo_cost', 'hi_cost', 'def_mm', 'sol', 'coeff']

# Caution: below mentioned features are strictly necessary, more features
# can be added but none should be removed from these. 
cfg.DATA = CN()

# Number of workers for data loader
cfg.DATA.DISK_DATA_ROOT = '/home/ahabbas/data/learnDBCA/cv_structure_pred/'
cfg.DATA.NUM_WORKERS = 0

# All datasets to be used in an experiment (training and testing):
# cfg.DATA.DATASETS = ['CT_SMALL', 'CT_LARGE'] # Underlying instances are of same size.
# cfg.DATA.TEST_FRACTION = [1.0, 0.0]

# cfg.DATA.CT_SMALL_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/cell-tracking-AISTATS-2020/small/', 'read_dual_converged' : True}) 
# cfg.DATA.CT_LARGE_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/cell-tracking-AISTATS-2020/large/', 'read_dual_converged' : True}) 

cfg.DATA.DATASETS = ['CT_TOY', 'CT_TOY2'] # Underlying instances are of same size.
cfg.DATA.TEST_FRACTION = [0.0, 1.0]

cfg.DATA.CT_TOY_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/cell-tracking-AISTATS-2020/toy/', 'read_dual_converged' : True}) 
cfg.DATA.CT_TOY2_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/cell-tracking-AISTATS-2020/toy2/', 'read_dual_converged' : True}) 

cfg.LOG_EVERY = 20
cfg.TRAIN = CN()
cfg.TRAIN.BASE_LR = 1e-4
cfg.TRAIN.BATCH_SIZE = 4
cfg.TRAIN.MAX_NUM_EPOCHS = 50
cfg.TRAIN.OPTIMIZER = "Adam"
cfg.TRAIN.NUM_ROUNDS = 3 # How many times rounding iterations.
cfg.TRAIN.NUM_DUAL_ITERATIONS = 20
cfg.TRAIN.DUAL_IMPROVEMENT_SLOPE = 1e-6
cfg.TRAIN.TRACK_GRAD_AFTER_ITR = 0
cfg.TRAIN.LOSS_DISCOUNT_FACTOR = 0.9

cfg.TEST = CN()
cfg.TEST.NUM_DUAL_ITERATIONS = 50
cfg.TEST.NUM_ROUNDS = 3 # How many times rounding iterations.
cfg.TEST.DUAL_IMPROVEMENT_SLOPE = 1e-6
cfg.TEST.BATCH_SIZE = 4
cfg.TEST.PERIOD = 5 # Validate after every n epoch (can be less than 1).
cfg.SEED = 1
cfg.OUTPUT_ROOT_DIR = '/home/ahabbas/projects/learnDBCA/out_primal/' # Do not change, if changed exclude it from sbatch files from copying.
cfg.OUT_REL_DIR = 'CT/v1/'

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return cfg.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`