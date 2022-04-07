from yacs.config import CfgNode as CN
import os

def get_all_lp_instances(root_dir, data_name, keyword = None):
    datasets = [data_name]
    files_to_load = []
    for path, subdirs, files in os.walk(root_dir):
        for instance_name in sorted(files):
            if not instance_name.endswith('.lp') or 'nan' in instance_name or 'normalized' in instance_name:
                continue

            if keyword is not None and not keyword in instance_name:
                continue

            files_to_load.append(instance_name)

    all_params = {}
    all_params[data_name + '_PARAMS'] = CN({
            'root_dir': root_dir, 
            'files_to_load': files_to_load,
            'read_dual_converged' : True,
            'need_gt': False})
    return datasets, all_params

cfg = CN()
cfg.DEVICE = 'gpu'

cfg.MODEL = CN()
cfg.MODEL.VAR_FEATURE_DIM = 16
cfg.MODEL.CON_FEATURE_DIM = 16
cfg.MODEL.EDGE_FEATURE_DIM = 8
cfg.MODEL.FEATURE_EXTRACTOR_DEPTH = 1
cfg.MODEL.PRIMAL_PRED_DEPTH = 1
cfg.MODEL.NUM_HIDDEN_LAYERS_EDGE = 1
cfg.MODEL.CKPT_PATH = None
cfg.MODEL.OMEGA = 0.5
cfg.MODEL.USE_LSTM_VAR = False
cfg.MODEL.USE_LAYER_NORM = False
cfg.MODEL.VAR_LP_FEATURES = ['orig_obj', 'deg', 'prev_obj']
cfg.MODEL.VAR_LP_FEATURES_INIT = ['obj', 'deg', 'obj']
cfg.MODEL.CON_LP_FEATURES = ['prev_lb', 'orig_lb', 'rhs', 'con_type', 'deg', 'orig_constant', 'prev_constant']
cfg.MODEL.CON_LP_FEATURES_INIT = ['lb', 'lb', 'rhs', 'con_type', 'deg', 'orig_constant', 'orig_constant']
cfg.MODEL.EDGE_LP_FEATURES = ['prev_sol', 'orig_sol', 'coeff', 'prev_mm_diff', 'orig_mm_diff']
cfg.MODEL.EDGE_LP_FEATURES_INIT = ['sol', 'sol', 'coeff', 'mm_diff', 'mm_diff']

cfg.DATA = CN()

# Number of workers for data loader
cfg.DATA.DISK_DATA_ROOT = '/home/ahabbas/data/learnDBCA/cv_structure_pred/'
cfg.DATA.NUM_WORKERS = 4

# All datasets to be used in an experiment (training and testing):
# cfg.DATA.DATASETS = ['CT_SMALL', 'CT_LARGE'] # Underlying instances are of same size.
# cfg.DATA.VAL_FRACTION = [1.0, 0.0]

# cfg.DATA.CT_SMALL_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/cell-tracking-AISTATS-2020/small/', 'read_dual_converged' : True}) 
# cfg.DATA.CT_LARGE_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/cell-tracking-AISTATS-2020/large/', 'read_dual_converged' : True}) 

# cfg.DATA.DATASETS = ['CT_LARGE', 'CT_SMALL'] #, 'GM_HOTEL', 'GM_HOUSE'] #, 'GM_WORMS_TRAIN', 'GM_WORMS_TEST']
# cfg.DATA.VAL_FRACTION = [0.0, 0.0] #, 0.0, 1.0]
# cfg.DATA.DATASETS = ['GM_HOTEL', 'GM_HOUSE'] #, 'GM_WORMS_TRAIN', 'GM_WORMS_TEST']
# cfg.DATA.VAL_FRACTION = [0.0, 0.0] #, 0.0, 1.0]
# cfg.DATA.CT_SMALL_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/cell-tracking-AISTATS-2020/small', 'read_dual_converged' : True}) 
# cfg.DATA.CT_LARGE_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/cell-tracking-AISTATS-2020/large', 'read_dual_converged' : True}) 
# cfg.DATA.GM_HOTEL_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/graph-matching/hotel_house/hotel/', 'read_dual_converged' : True}) 
# cfg.DATA.GM_HOUSE_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/graph-matching/hotel_house/house/', 'read_dual_converged' : True}) 
# cfg.DATA.GM_WORMS_TRAIN_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/graph-matching/worms/train_split/', 'read_dual_converged' : True}) 
# cfg.DATA.GM_WORMS_TEST_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/graph-matching/worms/test_split/', 'read_dual_converged' : True}) 

cfg.LOG_EVERY = 20
cfg.TRAIN = CN()
cfg.TRAIN.BASE_LR = 1e-2
cfg.TRAIN.LR_GAMMA = 0.1
cfg.TRAIN.BATCH_SIZE = 8
cfg.TRAIN.MAX_NUM_EPOCHS = 500
cfg.TRAIN.OPTIMIZER = "Adam"

cfg.TRAIN.NUM_ROUNDS = 20 # Max. possible number of rounding iterations.
cfg.TRAIN.NUM_ROUNDS_WITH_GRAD = 1 # Number of rounds in which gradients are backpropagated.
cfg.TRAIN.NUM_DUAL_ITERATIONS = 10
cfg.TRAIN.GRAD_DUAL_ITR_MAX_ITR = 3 # Gradient of dual iterations would be backpropagated for a maximum of last min(GRAD_DUAL_ITR_MAX_ITR, NUM_DUAL_ITERATIONS) many iterations.

cfg.TRAIN.DUAL_IMPROVEMENT_SLOPE = 1e-6
cfg.TRAIN.LOSS_DISCOUNT_FACTOR = 1.0
cfg.TRAIN.LOSS_MARGIN = 5e-3
cfg.TRAIN.START_EPISODIC_TRAINING_AFTER_EPOCH = 25
cfg.TRAIN.MIN_PERTURBATION = 1e-2
cfg.TRAIN.MM_AGR_LOSS_WEIGHT = 0.0
cfg.TRAIN.MM_TANH_MULT = 1e3

cfg.TEST = CN()
cfg.TEST.NUM_DUAL_ITERATIONS = 200
cfg.TEST.NUM_ROUNDS = 20 # How many times rounding iterations.
cfg.TEST.DUAL_IMPROVEMENT_SLOPE = 1e-6
cfg.TEST.VAL_BATCH_SIZE = 1
cfg.TEST.PERIOD = 50 # Validate after every n epoch (can be less than 1).

cfg.TEST.DATA = CN() # Stores dataset params used for testing only.

cfg.SEED = 1
cfg.OUTPUT_ROOT_DIR = '/home/ahabbas/projects/LearnDBCA/out_primal/' # Do not change, if changed exclude it from sbatch files from copying.
cfg.OUT_REL_DIR = 'CT/v3_test/'

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return cfg.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`