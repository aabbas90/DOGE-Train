from yacs.config import CfgNode as CN
from config_dual.defaults import get_cfg_defaults, get_all_lp_instances

cfg = get_cfg_defaults()

cfg.LOG_EVERY = 100

cfg.TRAIN.MAX_NUM_EPOCHS = 250
cfg.DATA.NUM_WORKERS = 4
cfg.DATA.DATASETS = ['GM_WORMS_TRAIN', 'GM_WORMS_VAL']
cfg.DATA.VAL_FRACTION = [0.0, 1.0]
WORM_TRAIN = [f'worm0{i}-16-03-11-1745.lp' for i in range(10)]
cfg.DATA.GM_WORMS_TRAIN_PARAMS = CN({'files_to_load': WORM_TRAIN, 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/full_inst/', 'read_dual_converged' : False}) 
#cfg.DATA.GM_WORMS_TRAIN_PARAMS = CN({'files_to_load': ['0.lp'], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/graph-matching/worms/train_split/worm01-16-03-11-1745/', 'read_dual_converged' : True}) 
cfg.DATA.GM_WORMS_VAL_PARAMS = CN({'files_to_load': ['1.lp', '2.lp'], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/graph-matching/worms/test_split/', 'read_dual_converged' : False}) 

cfg.TRAIN.BATCH_SIZE = 2
cfg.TRAIN.NUM_ROUNDS = 30
cfg.TRAIN.BASE_LR = 1e-3
cfg.TRAIN.USE_RELATIVE_GAP_LOSS = False

cfg.TEST.NUM_ROUNDS = 20
cfg.TEST.NUM_DUAL_ITERATIONS = 500
cfg.TEST.BATCH_SIZE = 1
cfg.TEST.VAL_PERIOD = 10000

test_datasets, test_params = get_all_lp_instances('/home/ahabbas/data/learnDBCA/cv_structure_pred/full_inst/', 'worms', 'worm', False)
cfg.TEST.DATA.DATASETS = test_datasets
cfg.TEST.DATA.update(test_params)

cfg.OUTPUT_ROOT_DIR = '/home/ahabbas/projects/LearnDBCA/out_dual/' # Do not change, if changed exclude it from sbatch files from copying.
cfg.OUT_REL_DIR = 'WORMS/v5/'