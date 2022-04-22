import ecole, os
import torch_geometric
import pickle
from tqdm import tqdm

n_rows = 50000
n_cols = 100000
density = 0.05
max_coeff = 100
seed = 0

def CreateSetCoverGenerator(n_rows, n_cols, density, max_coeff):
    return ecole.instance.SetCoverGenerator(n_rows, n_cols, density, max_coeff)

random_instance_generator = CreateSetCoverGenerator(n_rows, n_cols, density, max_coeff)
random_instance_generator.seed(seed)

ilp_ecole = next(random_instance_generator)
ilp_scip = ilp_ecole.as_pyscipopt()
ilp_scip.writeProblem('/home/ahabbas/data/learnDBCA/set_cover_random/test_3.lp')