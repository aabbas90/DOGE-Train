from ensurepip import version
from genericpath import isfile
import os, argparse
import numpy as np
import torch
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*if you want to see logs for the training epoch.*")
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
torch.use_deterministic_algorithms(False)
from config_dual.defaults import get_cfg_defaults
from data.dataloader import get_ilp_gnn_loaders
from dual_ascent import DualAscentBDD

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

def find_ckpt(ckpt_path, OUTPUT_ROOT_DIR):
    if os.path.isfile(ckpt_path):
        return ckpt_path
    versions = os.path.join(OUTPUT_ROOT_DIR, 'default')
    for folder in os.listdir(versions):
        possible_path = os.path.join(versions, folder, 'checkpoints/last.ckpt')
        if os.path.isfile(possible_path):
            print(f'Found checkpoint: {possible_path}')
            return possible_path
    return None

def main(args):
    print(datetime.now().time())
    cfg, output_dir = get_final_config(args)   
    seed_everything(cfg.SEED)
    gpus = 0
    if cfg.DEVICE == 'gpu':
        gpus = [0]
        # gpu_id = get_freer_gpu()
        # if gpu_id >= 0:
        #     print(f'Using GPU: {gpu_id}')
        #     gpus = [gpu_id]

    tb_logger = TensorBoardLogger(output_dir, default_hp_metric=False)
    ckpt_path = None
    if cfg.MODEL.CKPT_PATH is not None:
        ckpt_path = os.path.join(cfg.OUTPUT_ROOT_DIR, cfg.MODEL.CKPT_PATH)
        ckpt_path = find_ckpt(ckpt_path, cfg.OUTPUT_ROOT_DIR)

    assert ckpt_path is None or os.path.isfile(ckpt_path), f'CKPT: {ckpt_path} not found.'
    checkpoint_callback = ModelCheckpoint(save_last=True, every_n_epochs=cfg.TEST.VAL_PERIOD, save_top_k=1, monitor='val_loss')
    num_sanity_val_steps = 0
    if args.test_non_learned:
        num_sanity_val_steps = -1
    trainer = Trainer(deterministic=False,  # due to https://github.com/pyg-team/pytorch_geometric/issues/3175#issuecomment-1047886622
                    gpus = gpus,
                    max_epochs = cfg.TRAIN.MAX_NUM_EPOCHS, 
                    default_root_dir=output_dir,
                    check_val_every_n_epoch = cfg.TEST.VAL_PERIOD,
                    logger = tb_logger, 
                    resume_from_checkpoint = cfg.MODEL.CKPT_PATH, 
                    num_sanity_val_steps = num_sanity_val_steps, 
                    log_every_n_steps=cfg.LOG_EVERY,
                    track_grad_norm=2,
                    callbacks=[checkpoint_callback],
                    detect_anomaly = False)
                    # gradient_clip_val=100.0,
                    # gradient_clip_algorithm="norm",

    combined_train_loader, val_loaders, val_datanames, test_loaders, test_datanames = get_ilp_gnn_loaders(cfg, skip_dual_solved = True)
    if ckpt_path is not None:
        print(f'Loading checkpoint and hyperparameters from: {ckpt_path}')
        model = DualAscentBDD.load_from_checkpoint(ckpt_path,
            num_test_rounds = cfg.TEST.NUM_ROUNDS,
            num_dual_iter_test = cfg.TEST.NUM_DUAL_ITERATIONS,
            dual_improvement_slope_test = cfg.TEST.DUAL_IMPROVEMENT_SLOPE,
            val_datanames = val_datanames,
            test_datanames = test_datanames,
            non_learned_updates_test = args.test_non_learned)
    else:
        print(f'Initializing from scratch.')
        model = DualAscentBDD.from_config(cfg, 
            num_test_rounds = cfg.TEST.NUM_ROUNDS,
            num_dual_iter_test = cfg.TEST.NUM_DUAL_ITERATIONS,
            dual_improvement_slope_test = cfg.TEST.DUAL_IMPROVEMENT_SLOPE,
            val_datanames = val_datanames,
            test_datanames = test_datanames,
            non_learned_updates_test = args.test_non_learned)

    if not args.eval_only:
        trainer.fit(model, combined_train_loader, val_loaders)
        model.eval()
        trainer.test(model, test_dataloaders = test_loaders)
    else:
        model.eval()
        trainer.test(model, test_dataloaders = test_loaders)
    print(datetime.now().time())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--test-non-learned", action="store_true", help="Runs FastDOG updates.")
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