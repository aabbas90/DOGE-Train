from yacs.config import CfgNode as CN
from config_dual.defaults import get_cfg_defaults

cfg = get_cfg_defaults()

cfg.DATA.NUM_WORKERS = 4
cfg.DATA.DATASETS = ['CT_SMALL', 'CT_LARGE', 'GM_WORMS_TRAIN', 'GM_WORMS_TEST'] #, 'GM_HOTEL', 'GM_HOUSE']
cfg.DATA.TEST_FRACTION = [1.0, 0.0, 0.0, 1.0]
cfg.DATA.CT_SMALL_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/cell-tracking-AISTATS-2020/small', 'read_dual_converged' : True}) 
cfg.DATA.CT_LARGE_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/cell-tracking-AISTATS-2020/large', 'read_dual_converged' : True}) 
cfg.DATA.GM_HOTEL_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/graph-matching/hotel_house/hotel/', 'read_dual_converged' : True}) 
cfg.DATA.GM_HOUSE_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/graph-matching/hotel_house/house/', 'read_dual_converged' : True}) 
cfg.DATA.GM_WORMS_TRAIN_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/graph-matching/worms/train_split/', 'read_dual_converged' : True}) 
cfg.DATA.GM_WORMS_TEST_PARAMS = CN({'files_to_load': [], 'root_dir': '/home/ahabbas/data/learnDBCA/cv_structure_pred/graph-matching/worms/test_split/', 'read_dual_converged' : True}) 

cfg.OUTPUT_ROOT_DIR = '/home/ahabbas/projects/LearnDBCA/out_dual/' # Do not change, if changed exclude it from sbatch files from copying.
cfg.OUT_REL_DIR = 'CT/v1_conv_with_worms/'