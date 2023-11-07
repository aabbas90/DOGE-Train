# DOGE-Train: Discrete optimization on GPU with end-to-end training
A fast, scalable, data-driven approach for solving relaxations of 0-1 integer linear programs on GPU.
![DOGE pipeline](./data/doge_pipeline.PNG)

## Requirements
We use `Pytorch 2.0` and `CUDA 11.8`. `Gurobi` is used as one of the baselines and also for parsing ILPs. Consult `install.sh` for installing all requirements.

## Datasets:
### 1. Synthetic problems:
First generate synthetic problems through the following command:
```bash
python data/generate_independent_set_inst.py
```
This will generate independent set problem instances as used in the paper and write them in `datasets/MIS` directory. 

### 2. Custom datasets:
For other datasets modify the variable `DATA_DIR` in config files appropriately. The format should be 
```
./datasets/<DATA_NAME>/train_split/instances/<training .lp files here>
./datasets/<DATA_NAME>/test_split/instances/<testing .lp files here>
```

See `configs/config_mis.py` for an example. Config files for other datasets used in the paper are provided in the `configs/` folder.

## Training:
For synthetic independent set problems generated above run `train_mis.sh` script. For details about configs, command-line parameters see `configs/config_mis.py`,`configs/default` and `train_doge.py`.
Note that for testing we automatically run Gurobi for comparison. This can be disabled by setting `need_gt = False` in `configs/config_mis.py`. 

## Code organization:

- `train_doge.py`: Entry point. 
- `doge.py`: Code for training and testing. Uses `pytorch lightning` for ease of use. 
- `configs/`: Contains configuration files for all datasets. 
- `data/`: Contains code related to ILP reading, processing etc. 
- `model/model.py`: Contains the neural network related code. 
- `model/solver_utils.py`: Provides interface to the parallel deferred min-marginal averaging algorithm.   
- `external/`: Contains external dependencies. 
