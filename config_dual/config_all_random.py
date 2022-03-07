from yacs.config import CfgNode as CN

cfg = CN()

cfg.MODEL = CN()
cfg.MODEL.VAR_FEATURE_DIM = 32
cfg.MODEL.CON_FEATURE_DIM = 32
cfg.MODEL.EDGE_FEATURE_DIM = 8
cfg.MODEL.CKPT_PATH = None

cfg.DATA = CN()

cfg.DATA.VAR_FIXED_FEATURES = ['obj', 'deg']
cfg.DATA.CON_FIXED_FEATURES = ['rhs', 'leq', 'geq', 'lb', 'deg']

# CapacitatedFacilityLocation problem has non-binary variables.
cfg.DATA.DATASETS = ['SetCover_Random', 'IndependentSet_Random', 'CombinatorialAuction_Random']
cfg.DATA.TEST_FRACTION = [0.2, 0.8, 0.2] 
cfg.DATA.SetCover_Random_PARAMS = CN({'num_samples': 1024, 'n_rows': 500, 'n_cols': 1000, 'density': 0.05, 'max_coeff': 100})
cfg.DATA.IndependentSet_Random_PARAMS = CN({'num_samples': 1024, 'n_nodes': 500, 'edge_probability': 0.25, 'affinity': 4})
cfg.DATA.CombinatorialAuction_Random_PARAMS = CN({'num_samples': 1024})

cfg.TRAIN = CN()
cfg.TRAIN.BATCH_SIZE = 32
cfg.TRAIN.MAX_NUM_EPOCHS = 250

cfg.TEST = CN()
cfg.TEST.NUM_ITERATIONS = 500
cfg.TEST.PERIOD = 25 # Validate after every n epochs

cfg.OUT_REL_DIR = 'IS/v1/'