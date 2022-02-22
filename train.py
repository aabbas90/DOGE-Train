import os, torch
from config.defaults import get_cfg_defaults
import argparse 
import numpy as np
from data.dataloader import get_ilp_gnn_loaders
from model.solver_utils import init_solver_and_get_states, compute_per_bdd_lower_bound

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

def main(args):
    cfg, output_dir = get_final_config(args)
    device = torch.device("cuda:0")
    combined_train_loader, test_loaders, test_datanames = get_ilp_gnn_loaders(cfg)
    batch = next(iter(combined_train_loader))
    solvers, lo_costs, hi_costs, def_mm = init_solver_and_get_states(batch, device, num_iterations = 100, omega = 0.5)
    lb_per_bdd_batch = compute_per_bdd_lower_bound(solvers, lo_costs, hi_costs)

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