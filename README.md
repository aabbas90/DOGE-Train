# DOGE-Train: Discrete optimization on GPU with end-to-end training
A fast, scalable, data-driven approach for solving relaxations of 0-1 integer linear programs on GPU.

## Requirements
We use `Pytorch 2.0` and `CUDA 11.8`. `Gurobi` is used as one of the baselines and also for parsing ILPs. Consult `install.sh` for installing all requirements.

## Training on random problems

### 1. Data generation:
The following command:
```bash
python data/generate_independent_set_inst.py
```
will generate independent set problem instances as used in the paper and write them in `datasets/MIS` directory. 

### 2. Training and evaluation:
Run `train_mis.sh` script. For details about configs, command-line parameters see `configs/config_mis.py`,`configs/default` and `train_doge.py`.

## Code organization:

- `train_doge.py`: Entry point. 
- `doge.py`: Code for training and testing. Uses `pytorch lightning` for ease of use. 
- `configs/`: Contains configuration files for all datasets. 
- `data/`: Contains code related to ILP reading, processing etc. 
- `model/model.py`: Contains the neural network related code. 
- `model/solver_utils.py`: Provides interface to the parallel deferred min-marginal averaging algorithm.   
- `external/`: Contains external dependencies. 