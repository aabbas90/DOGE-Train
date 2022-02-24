import os, argparse
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*if you want to see logs for the training epoch.*")
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
torch.use_deterministic_algorithms(False)
from config_primal.defaults import get_cfg_defaults
from data.dataloader import get_ilp_gnn_loaders
from primal_rounding import PrimalRoundingBDD

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

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    if len(memory_available) == 0:
        return -1
    return np.argmax(memory_available)

def get_latest_checkpoint(output_dir):
    rel_path = 'default/version_0/checkpoints/last.ckpt'
    p = os.path.join(output_dir, rel_path)
    assert os.path.exists(p), f"Automatic checkpoint detector could not find at default path: {p}."
    print(f"Found checkpoint: {p} for evaluation.")
    return p

def main(args):
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
    assert cfg.MODEL.CKPT_PATH is None or os.path.isfile(cfg.MODEL.CKPT_PATH), "CKPT not found."
    checkpoint_callback = ModelCheckpoint(save_last=True)
    trainer = Trainer(deterministic=False,  # due to https://github.com/pyg-team/pytorch_geometric/issues/3175#issuecomment-1047886622
                    gpus = gpus,
                    max_epochs = cfg.TRAIN.MAX_NUM_EPOCHS, 
                    default_root_dir=output_dir,
                    check_val_every_n_epoch = cfg.TEST.PERIOD,
                    logger = tb_logger, 
                    resume_from_checkpoint = cfg.MODEL.CKPT_PATH, 
                    num_sanity_val_steps=0, 
                    log_every_n_steps=cfg.LOG_EVERY,
                    callbacks=[checkpoint_callback])

    combined_train_loader, test_loaders, test_datanames = get_ilp_gnn_loaders(cfg)
    ckpt_path = cfg.MODEL.CKPT_PATH
    if args.eval_only and ckpt_path is None:
        ckpt_path = get_latest_checkpoint(os.path.join(cfg.OUTPUT_ROOT_DIR, cfg.OUT_REL_DIR))

    if ckpt_path is not None:
        print(f'Loading checkpoint and hyperparameters from: {ckpt_path}')
        model = PrimalRoundingBDD.load_from_checkpoint(ckpt_path)
        model.test_datanames = test_datanames
    else:
        print(f'Initializing from scratch.')
        model = PrimalRoundingBDD.from_config(cfg, test_datanames)

    if not args.eval_only:
        trainer.fit(model, combined_train_loader, test_loaders)
    model.eval()
    trainer.test(model, test_dataloaders = test_loaders)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
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