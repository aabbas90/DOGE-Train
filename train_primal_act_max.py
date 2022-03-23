import os, argparse
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*if you want to see logs for the training epoch.*")
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
import torch
torch.use_deterministic_algorithms(False)
from config_primal.defaults import get_cfg_defaults
from data.dataloader import get_ilp_gnn_loaders
from primal_act_max import PrimalActMax

def get_final_config(args):
    cfg = get_cfg_defaults()
    if (hasattr(args, 'config_file')) and os.path.exists(args.config_file):
        cfg.set_new_allowed(True)
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    output_dir = os.path.join(cfg.OUTPUT_ROOT_DIR, cfg.OUT_REL_DIR)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "config.yaml")
    with open(path, 'w') as yaml_file:
        cfg.dump(stream = yaml_file, default_flow_style=False)

    print('USING FOLLOWING CONFIG:')
    print(cfg)
    print("Wrote config file at: {}".format(path))
    return cfg, output_dir

def main(args):
    cfg, output_dir = get_final_config(args)   
    seed_everything(cfg.SEED)
    gpus = 0
    if cfg.DEVICE == 'gpu':
        gpus = [0]

    tb_logger = TensorBoardLogger(output_dir, default_hp_metric=False)

    trainer = Trainer(deterministic=False,  # due to https://github.com/pyg-team/pytorch_geometric/issues/3175#issuecomment-1047886622
                    gpus = gpus,
                    max_epochs = 0, 
                    default_root_dir=output_dir,
                    logger = tb_logger, 
                    num_sanity_val_steps=-1,
                    log_every_n_steps=cfg.LOG_EVERY,
                    detect_anomaly = False)

    combined_train_loader, val_loaders, val_datanames, test_loaders, test_datanames = get_ilp_gnn_loaders(cfg)
    model = PrimalActMax.from_config(cfg, 
            test_datanames = test_datanames, 
            num_dual_iter_test = cfg.TEST.NUM_DUAL_ITERATIONS,
            num_test_rounds = cfg.TEST.NUM_ROUNDS,
            dual_improvement_slope_test = cfg.TEST.DUAL_IMPROVEMENT_SLOPE)

    model.eval()
    trainer.test(model, test_dataloaders = test_loaders)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. ",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    print("Command Line Args:")
    print(args)
    main(args)