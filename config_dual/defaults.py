from yacs.config import CfgNode as CN

cfg = CN()
cfg.DEVICE = 'gpu'

cfg.MODEL = CN()
cfg.MODEL.VAR_FEATURE_DIM = 32
cfg.MODEL.CON_FEATURE_DIM = 32
cfg.MODEL.EDGE_FEATURE_DIM = 16
cfg.MODEL.FEATURE_EXTRACTOR_DEPTH = 5
cfg.MODEL.BDD_LAYER_DEPTH = 1
cfg.MODEL.CKPT_PATH = None

# Caution: below mentioned features are strictly necessary, more features
# can be added but none should be removed from these. 
cfg.DATA = CN()
cfg.DATA.VAR_FIXED_FEATURES = ['obj', 'deg']
cfg.DATA.CON_FIXED_FEATURES = ['rhs', 'leq', 'geq', 'lb', 'deg']
cfg.DATA.EDGE_FIXED_FEATURES = ['mm_0', 'mm_1', 'sol', 'coeff']

# Number of workers for data loader
cfg.DATA.RANDOM_DATA_ROOT = '/home/ahabbas/data/learnDBCA/random_instances_with_lp/'
cfg.DATA.DISK_DATA_ROOT = ''
cfg.DATA.NUM_WORKERS = 2

# All datasets to be used in an experiment (training and testing):
cfg.DATA.DATASETS = ['SetCover_Random', 'IndependentSet_Random']
cfg.DATA.TEST_FRACTION = [0.2, 0.8] 
cfg.DATA.SetCover_Random_PARAMS = CN({'num_samples': 1024, 'n_rows': 500, 'n_cols': 1000, 'density': 0.05, 'max_coeff': 100})
cfg.DATA.IndependentSet_Random_PARAMS = CN({'num_samples': 1024, 'n_nodes': 500, 'edge_probability': 0.25, 'affinity': 4})
cfg.DATA.CapacitatedFacilityLocation_Random_PARAMS = CN({'num_samples': 1024})
cfg.DATA.CombinatorialAuction_Random_PARAMS = CN({'num_samples': 1024})

cfg.DATA.RAIL01_PARAMS = CN({'files_to_load': ['rail01.lp'], 'root_dir': '/BS/discrete_opt/nobackup/miplib_collection/lp_format_presolved/'}) 
cfg.DATA.RAIL02_PARAMS = CN({'files_to_load': ['rail02.lp'], 'root_dir': '/BS/discrete_opt/nobackup/miplib_collection/lp_format_presolved/'}) 

cfg.LOG_EVERY = 20
cfg.TRAIN = CN()
cfg.TRAIN.BASE_LR = 1e-4
cfg.TRAIN.BATCH_SIZE = 32
cfg.TRAIN.MAX_NUM_EPOCHS = 50
cfg.TRAIN.OPTIMIZER = "Adam"
cfg.TRAIN.START_STEP_RANGE = [1, 15]
cfg.TRAIN.NUM_ITERATIONS = 5
cfg.TRAIN.BACKPROP_MM_FINITE_DIFF = False
cfg.TRAIN.BACKPROP_MM_FINITE_DIFF_STEP_SIZE = 1.0

cfg.TEST = CN()
cfg.TEST.BATCH_SIZE = 32
cfg.TEST.NUM_ITERATIONS = 1000
cfg.TEST.PERIOD = 5 # Validate after every n epoch (can be less than 1).
cfg.SEED = 1
cfg.OUTPUT_ROOT_DIR = '/home/ahabbas/projects/learnDBCA/out/' # Do not change, if changed exclude it from sbatch files from copying.
cfg.OUT_REL_DIR = 'experiments_with_lp_sol/v4_SC/'
cfg.TEST.BDD_SOLVER_EXEC = '/home/ahabbas/projects/BDD/build_debug_ninja/src/bdd_solver_cl'

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return cfg.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`