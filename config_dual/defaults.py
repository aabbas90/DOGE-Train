from yacs.config import CfgNode as CN
import os

def get_all_lp_instances(root_dir, data_name, keyword = None, read_converged = False):
    datasets = [data_name]
    files_to_load = []
    for path, subdirs, files in os.walk(root_dir):
        for instance_name in sorted(files):
            if not instance_name.endswith('.lp') or 'nan' in instance_name or 'normalized' in instance_name:
                continue
            
            if keyword is None or not keyword in instance_name:
                continue

            files_to_load.append(instance_name)

    all_params = {}
    all_params[data_name + '_PARAMS'] = CN({
            'root_dir': root_dir, 
            'files_to_load': files_to_load,
            'read_dual_converged' : read_converged,
            'need_gt': False})
    return datasets, all_params

cfg = CN()
cfg.DEVICE = 'gpu'

cfg.MODEL = CN()
cfg.MODEL.FULL_COORDINATE_ASCENT = False
cfg.MODEL.PREDICT_OMEGA = False
cfg.MODEL.VAR_FEATURE_DIM = 16
cfg.MODEL.CON_FEATURE_DIM = 16
cfg.MODEL.EDGE_FEATURE_DIM = 8
cfg.MODEL.FEATURE_EXTRACTOR_DEPTH = 1
cfg.MODEL.DUAL_PRED_DEPTH = 1
cfg.MODEL.CKPT_PATH = None
cfg.MODEL.OMEGA_INITIAL = 0.5
cfg.MODEL.USE_LAYER_NORM = True
cfg.MODEL.VAR_LP_FEATURES = ['obj', 'deg']
cfg.MODEL.VAR_LP_FEATURES_INIT = ['obj', 'deg']
cfg.MODEL.CON_LP_FEATURES = ['lb', 'rhs', 'con_type', 'deg', 'prev_lb'] # 'round_index',
cfg.MODEL.CON_LP_FEATURES_INIT = ['lb', 'rhs', 'con_type', 'deg', 'prev_lb'] #'round_index',
cfg.MODEL.EDGE_LP_FEATURES = ['sol', 'coeff', 'dist_weights', 'prev_sol', 'omega'] #, 'grad_dist_weights', 'grad_omega']
cfg.MODEL.EDGE_LP_FEATURES_INIT = ['sol', 'coeff', 'dist_weights', 'prev_sol', 'omega'] #, 'grad_dist_weights', 'grad_omega']
cfg.MODEL.NUM_HIDDEN_LAYERS_EDGE = 0
cfg.MODEL.USE_NET_SOLVER_COSTS = False

cfg.DATA = CN()

# Number of workers for data loader
cfg.DATA.DISK_DATA_ROOT = '/home/ahabbas/data/learnDBCA/cv_structure_pred/'
cfg.DATA.NUM_WORKERS = 0

cfg.DATA.DATASETS = ['CT_SMALL', 'CT_SMALL'] #'CT_LARGE'] #, 'GM_WORMS_TRAIN', 'GM_WORMS_TEST'] #, 'GM_HOTEL', 'GM_HOUSE']
cfg.DATA.VAL_FRACTION = [1.0, 0.0] #, 0.0, 1.0]
cfg.DATA.CT_SMALL_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/cell-tracking-AISTATS-2020/small/drosophila/', 'read_dual_converged' : False}) 
#cfg.DATA.CT_SMALL_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/cell-tracking-AISTATS-2020/small', 'read_dual_converged' : False}) 
cfg.DATA.CT_LARGE_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/cell-tracking-AISTATS-2020/large', 'read_dual_converged' : False}) 
cfg.DATA.GM_HOTEL_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/graph-matching/hotel_house/hotel/', 'read_dual_converged' : False}) 
cfg.DATA.GM_HOUSE_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/graph-matching/hotel_house/house/', 'read_dual_converged' : False}) 
cfg.DATA.GM_WORMS_TRAIN_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/graph-matching/worms/train_split/', 'read_dual_converged' : False}) 
cfg.DATA.GM_WORMS_TEST_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/graph-matching/worms/test_split/', 'read_dual_converged' : False}) 

cfg.TRAIN = CN()
cfg.TRAIN.BASE_LR = 1e-4
cfg.TRAIN.BATCH_SIZE = 8
cfg.TRAIN.MAX_NUM_EPOCHS = 300
cfg.TRAIN.OPTIMIZER = "Adam"
cfg.TRAIN.USE_RELATIVE_GAP_LOSS = False

cfg.TRAIN.NUM_ROUNDS = 1 # Max. possible number of dual iteration rounds.
cfg.TRAIN.NUM_ROUNDS_WITH_GRAD = 1 # Number of rounds in which gradients are backpropagated.
cfg.TRAIN.NUM_DUAL_ITERATIONS = 10
cfg.TRAIN.GRAD_DUAL_ITR_MAX_ITR = 10 # Gradient of dual iterations would be backpropagated for a maximum of last min(GRAD_DUAL_ITR_MAX_ITR, NUM_DUAL_ITERATIONS) many iterations.

cfg.TRAIN.DUAL_IMPROVEMENT_SLOPE = 1e-6
cfg.TRAIN.LOSS_DISCOUNT_FACTOR = 1.0
cfg.TRAIN.LOSS_MARGIN = 5e-3
cfg.TRAIN.START_EPISODIC_TRAINING_AFTER_EPOCH = 25

cfg.TEST = CN()
cfg.TEST.NUM_DUAL_ITERATIONS = 10
cfg.TEST.NUM_ROUNDS = 1 # How many times dual iterations. #TODOAA: Implement break if slope < improvement.
cfg.TEST.DUAL_IMPROVEMENT_SLOPE = 1e-6
cfg.TEST.VAL_BATCH_SIZE = 1
cfg.TEST.PERIOD = 50 # Validate after every n epoch (can be less than 1).

cfg.TEST.DATA = CN() # Stores dataset params used for testing only.
cfg.SEED = 1
cfg.OUTPUT_ROOT_DIR = '/home/ahabbas/projects/LearnDBCA/out_dual/' # Do not change, if changed exclude it from sbatch files from copying.
cfg.OUT_REL_DIR = 'CT/v1/'

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return cfg.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`