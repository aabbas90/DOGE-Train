import pickle, os
import bdd_cuda_torch
from torch_scatter import scatter_sum, scatter_softmax, scatter_mean, scatter_add, scatter_std
import BDD.ILP_instance_py as ilp_instance_bbd
import BDD.bdd_cuda_learned_mma_py as bdd_solver
import torch
import numpy as np

def print_stats(t, name):
    print(f'{name}') 
    if (torch.abs(t).max() > 1e6 or torch.any(~torch.isfinite(t))):
        print(f'\t min={t.min():.3f}, max={t.max():.3f}, mean: {t.mean():.3f}, std: {t.std():.3f}')

def dual_iterations(solvers, solver_state, dist_weights, num_iterations, omega, improvement_slope = 1e-6, grad_dual_itr_max_itr = None, logger = None, filepahts = None, step = None):
    if grad_dual_itr_max_itr is None:
        grad_dual_itr_max_itr = num_iterations
    num_caches = min(int(grad_dual_itr_max_itr / 3), 25)
    lo_costs, hi_costs, def_mm = bdd_cuda_torch.DualIterations.apply(solvers, solver_state['lo_costs'], solver_state['hi_costs'], solver_state['def_mm'], 
                                                                    dist_weights, num_iterations, omega, grad_dual_itr_max_itr, improvement_slope, num_caches,
                                                                    logger, filepahts, step)
    solver_state['lo_costs'] = lo_costs
    solver_state['hi_costs'] = hi_costs
    solver_state['def_mm'] = def_mm
    return solver_state # updated solver_state

#bdd_repr = pickle.load(open('/home/ahabbas/data/learnDBCA/miplib_crops/easy/instances/academictimetablesmall/instances/3_subset_2_bdd_repr.pkl', 'rb'))
#bdd_repr = pickle.load(open('/home/ahabbas/data/learnDBCA/miplib_crops/easy/instances/academictimetablesmall/instances/3_bdd_repr_nan.pkl', 'rb'))

root_dir = '/home/ahabbas/data/learnDBCA/miplib_crops/easy/instances/academictimetablesmall/instances/'
#root_dir = '/home/ahabbas/data/learnDBCA/cv_structure_pred/graph-matching/worms/train_split/'
num_itr = 100
num_grad_itr = 100

for path, subdirs, files in os.walk(root_dir):
    for instance_name in sorted(files):
        if not 'bdd_repr_dual_converged.pkl' in instance_name:
            continue
        instance_path = os.path.join(path, instance_name)

        bdd_repr = pickle.load(open(instance_path, 'rb'))
        solver = pickle.loads(bdd_repr['solver_data'])

        lo_costs = torch.empty((solver.nr_layers()), device = 'cuda', dtype = torch.float32)
        hi_costs = torch.empty_like(lo_costs)
        def_mm = torch.zeros_like(lo_costs)
            
        omega = torch.ones((solver.nr_layers()), device = 'cuda', dtype = torch.float32) - 0.5
        dist_weights = torch.ones((solver.nr_layers()), device = 'cuda', dtype = torch.float32)
        var_indices = torch.from_numpy(bdd_repr['var_indices']).to(omega.device).to(torch.long)
        con_indices = torch.from_numpy(bdd_repr['con_indices']).to(omega.device).to(torch.long)
        dist_weights = scatter_softmax(dist_weights, var_indices)

        #solver.get_solver_costs(lo_costs.data_ptr(), hi_costs.data_ptr(), def_mm.data_ptr())

        # lo_costs = lo_costs * 0 + 1
        # hi_costs = hi_costs * 0 + 1

        solver.set_solver_costs(lo_costs.data_ptr(), hi_costs.data_ptr(), def_mm.data_ptr())
        num_act = solver.iterations(dist_weights.data_ptr(), num_itr, 1.0, 0.0, omega.data_ptr(), True)
        assert num_itr == num_act

        grad_lo_costs = torch.ones((solver.nr_layers()), device = 'cuda', dtype = torch.float32)
        grad_hi_costs = torch.ones_like(grad_lo_costs)
        grad_def_mm = torch.ones_like(grad_lo_costs)
        grad_dist_weights = torch.zeros_like(grad_lo_costs)
        grad_omega = torch.zeros_like(grad_lo_costs)
        solver.grad_iterations(dist_weights.data_ptr(), grad_lo_costs.data_ptr(), grad_hi_costs.data_ptr(),
                                grad_def_mm.data_ptr(), grad_dist_weights.data_ptr(), grad_omega.data_ptr(),
                                1.0, 0, num_itr, omega.data_ptr(), True, 10)
        print_stats(grad_lo_costs, f"grad_lo_costs of {instance_path}")

        # large_edges = [ 1075,  2154,  3289,  4394,  5495,  6510,  7581,  8465,  9424, 10033,
        #         10777, 11344, 11847, 12352, 12853, 13354, 13855, 14336, 14490, 14971,
        #         15105, 15228, 15362, 15485, 15608, 15731, 15854, 15977, 16070, 16163,
        #         16256, 16340] # of constraint 1030.

        # print(f'\n\n\n\n\n {num_itr}')
        # print(grad_lo_costs[large_edges])

        # grad_var = scatter_mean(torch.abs(grad_lo_costs), var_indices)
        # grad_con = scatter_mean(torch.abs(grad_lo_costs), con_indices)
        # print(torch.where(grad_var > 6290483712))
        # print(torch.where(grad_con > 6290483712))
        # print_stats(grad_var, "grad_var")

        # print_stats(grad_con, "grad_con")

        # mm_diff = torch.zeros_like(lo_costs)
        # solver.all_min_marginal_differences(mm_diff.data_ptr())
