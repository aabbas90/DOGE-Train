from yacs.config import CfgNode as CN
from config_primal.defaults import get_cfg_defaults

cfg = get_cfg_defaults()

cfg.DATA.DATASETS = ['CT_LARGE', 'CT_SMALL', 'GM_WORMS_TRAIN', 'GM_WORMS_TEST'] #'GM_HOTEL', 'GM_HOUSE'] #,
cfg.DATA.TEST_FRACTION = [0.0, 1.0, 0.0, 1.0] #, 0.0, 1.0]

# cfg.DATA.DATASETS = ['GM_WORMS_TRAIN', 'GM_WORMS_TEST']
# cfg.DATA.TEST_FRACTION = [0.0, 0.0]
cfg.DATA.GM_WORMS_TRAIN_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/graph-matching/worms/train_split/', 'read_dual_converged' : True}) 
cfg.DATA.GM_WORMS_TEST_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/graph-matching/worms/test_split/', 'read_dual_converged' : True}) 

cfg.TRAIN.BATCH_SIZE = 12
cfg.TRAIN.MAX_NUM_EPOCHS = 200
cfg.TRAIN.START_EPISODIC_TRAINING_AFTER_EPOCH = 20