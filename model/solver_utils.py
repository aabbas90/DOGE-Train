import torch
import pickle
import BDD.bdd_cuda_learned_mma_py as bdd_solver
from torch_scatter import scatter_sum, scatter_softmax
import bdd_cuda_torch

def init_solver_and_get_states(batch, device, gt_solution_type, num_iterations = 0, improvement_slope = 1e-6, omega = 0.5):
    num_edges = batch.edge_index_var_con.shape[1]
    lo_costs_out = torch.empty((num_edges), device = device, dtype = torch.float32)
    hi_costs_out = torch.empty_like(lo_costs_out)
    def_mm_out = torch.empty_like(lo_costs_out)
    solvers = []
    layer_start = 0
    for b in range(len(batch.solver_data)):
        solver = pickle.loads(batch.solver_data[b])
        batch.solver_data[b] = None # Free-up the memory.
        # Run non learned solver for num_iterations e.g. to take it to states where optimization becomes difficult.
        solver.non_learned_iterations(omega, num_iterations, improvement_slope)
        solver.get_solver_costs(lo_costs_out[layer_start].data_ptr(), hi_costs_out[layer_start].data_ptr(), def_mm_out[layer_start].data_ptr())
        layer_start += solver.nr_layers()
        solvers.append(solver)

    assert(layer_start == lo_costs_out.shape[0])

    solver_state = {'lo_costs': lo_costs_out, 'hi_costs': hi_costs_out, 'def_mm': def_mm_out}
    per_bdd_lb = compute_per_bdd_lower_bound(solvers, solver_state)
    per_bdd_sol = compute_per_bdd_solution(solvers, solver_state)
    dist_weights = normalize_distribution_weights(torch.ones_like(lo_costs_out), batch.edge_index_var_con)
    valid_edge_mask_list = get_valid_edge_mask(batch, solvers)
    if gt_solution_type is not None:
        gt_sol_var, gt_sol_edge = get_valid_target_solution_edge(solvers, batch.num_vars, batch.edge_index_var_con[0], batch.gt_info, valid_edge_mask_list, gt_solution_type)
    else:
        gt_sol_var = None
        gt_sol_edge = None
    valid_edge_mask = torch.cat(valid_edge_mask_list, 0)
    return solvers, solver_state, per_bdd_sol, per_bdd_lb, dist_weights, valid_edge_mask, gt_sol_edge, gt_sol_var

def non_learned_updates(batch, num_iterations = 0, improvement_slope = 1e-6, omega = 0.5):
    layer_start = 0
    for (b, solver) in enumerate(batch.solvers):  
        solver.set_solver_costs(batch.solver_state['lo_costs'][layer_start].data_ptr(), 
                                batch.solver_state['hi_costs'][layer_start].data_ptr(), 
                                batch.solver_state['def_mm'][layer_start].data_ptr())
        solver.non_learned_iterations(omega, num_iterations, improvement_slope)
        solver.get_solver_costs(batch.solver_state['lo_costs'][layer_start].data_ptr(), 
                                batch.solver_state['hi_costs'][layer_start].data_ptr(), 
                                batch.solver_state['def_mm'][layer_start].data_ptr())
        layer_start += solver.nr_layers()
    
    # Update lower bounds:
    batch.con_lp_f[:, 0] = compute_per_bdd_lower_bound(batch.solvers, batch.solver_state) 
    # Update LP feature information so that GNN can be run afterwards.
    batch.edge_rest_lp_f[:, 0] = compute_per_bdd_solution(batch.solvers, batch.solver_state)
    batch.edge_rest_lp_f[:, 2] = normalize_distribution_weights(torch.ones_like(batch.solver_state['lo_costs']), batch.edge_index_var_con)
    batch.solver_state['def_mm'][~batch.valid_edge_mask] = 0 # Locations of terminal nodes can contain nans.
    return batch

# Find edges which correspond to valid primal variables (as each BDD contains 1 invalid primal variable each.)
def get_valid_target_solution_edge(solvers, num_vars, var_indices, gt_info, valid_edge_mask_list, solution_type):
    assert(solution_type == 'lp_stats' or solution_type == 'ilp_stats')
    gt_sol_var = []
    gt_sol_edge = []
    layer_start = 0
    var_offset = 0
    for b in range(len(num_vars)):
        sol = gt_info[solution_type]['sol'][b]
        if sol is not None:
            current_var_sol = torch.from_numpy(sol).to(torch.float32).to(var_indices.device)
            gt_sol_var.append(current_var_sol)
            current_var_indices = var_indices[layer_start: layer_start + solvers[b].nr_layers()] - var_offset
            current_edge_sol = current_var_sol[current_var_indices]
            current_edge_sol_valid = current_edge_sol[valid_edge_mask_list[b]]
            gt_sol_edge.append(current_edge_sol_valid)
        else:
            gt_sol_var.append(None)
            gt_sol_edge.append(None)
        layer_start += solvers[b].nr_layers()
        var_offset += num_vars[b]
    return gt_sol_var, gt_sol_edge 

def get_valid_edge_mask(batch, solvers):
    masks_list = [] #torch.ones(batch.num_edges, dtype=torch.bool)
    for (b, solver) in enumerate(solvers):
        terminal_indices = torch.empty((solver.nr_bdds()), device = batch.edge_index_var_con.device, dtype = torch.int32)
        solver.terminal_layer_indices(terminal_indices.data_ptr())
        current_mask = torch.ones(solver.nr_layers(), dtype = torch.bool)
        current_mask[terminal_indices.to(torch.long)] = False
        masks_list.append(current_mask)
    return masks_list

def compute_per_bdd_lower_bound(solvers, solver_state):
    return bdd_cuda_torch.ComputeLowerBoundperBDD.apply(solvers, solver_state['lo_costs'], solver_state['hi_costs'])

def compute_per_bdd_solution(solvers, solver_state):
    sol = bdd_cuda_torch.ComputePerBDDSolutions(solvers, solver_state['lo_costs'], solver_state['hi_costs'])
    return sol

def distribute_delta(solvers, solver_state):
    lo_costs, hi_costs = bdd_cuda_torch.DistributeDeferredDelta.apply(solvers, solver_state['lo_costs'], solver_state['hi_costs'], solver_state['def_mm'])
    return {'lo_costs': lo_costs, 'hi_costs': hi_costs, 'def_mm': torch.zeros_like(solver_state['def_mm'])}

def dual_iterations(solvers, solver_state, dist_weights, num_iterations, omega, improvement_slope = 1e-6, grad_dual_itr_max_itr = None):
    if grad_dual_itr_max_itr is None:
        grad_dual_itr_max_itr = num_iterations
    lo_costs, hi_costs, def_mm = bdd_cuda_torch.DualIterations.apply(solvers, solver_state['lo_costs'], solver_state['hi_costs'], solver_state['def_mm'], 
                                                                    dist_weights, num_iterations, omega, grad_dual_itr_max_itr, improvement_slope)
    solver_state['lo_costs'] = lo_costs
    solver_state['hi_costs'] = hi_costs
    solver_state['def_mm'] = def_mm
    return solver_state # updated solver_state

def compute_all_min_marginal_diff(solvers, solver_state):
    mm_diff = bdd_cuda_torch.ComputeAllMinMarginalsDiff.apply(solvers, solver_state['lo_costs'], solver_state['hi_costs'])
    return mm_diff

def normalize_distribution_weights(dist_weights, edge_index_var_con):
# Normalize distribution weights so that they sum upto 1 for each variable.
    var_indices = edge_index_var_con[0, :]
    dist_weights_sum = scatter_sum(dist_weights, var_indices)[var_indices]
    return dist_weights / dist_weights_sum

def normalize_distribution_weights_softmax(dist_weights, edge_index_var_con):
    var_indices = edge_index_var_con[0, :]
    softmax_scores = scatter_softmax(dist_weights, var_indices)
    return softmax_scores

# def perturb_primal_costs(solvers, lo_costs, hi_costs, primal_perturbation):
#     primal_pert_lo = torch.relu(-primal_perturbation)
#     primal_pert_hi = torch.relu(primal_perturbation)
#     lo_costs, hi_costs = bdd_cuda_torch.PerturbPrimalCosts.apply(solvers, primal_pert_lo, primal_pert_hi, lo_costs, hi_costs)
#     return lo_costs, hi_costs

def perturb_primal_costs(lo_costs, hi_costs, primal_perturbation, edge_index_var_con):
    dist_weights = normalize_distribution_weights(torch.ones_like(lo_costs), edge_index_var_con)
    primal_perturbation_edge = primal_perturbation[edge_index_var_con[0]] * dist_weights
    lo_costs = lo_costs + torch.relu(-primal_perturbation_edge)
    hi_costs = hi_costs + torch.relu(primal_perturbation_edge)
    return lo_costs, hi_costs

# Assumes that dual update is performed using only hi_costs and lo_costs remain 0.
# Finds weights w s.t. dual_cost_ij = w_ij * c_i => w_ij = dual_cost_ij / c_i for variable i and constraint j.
def compute_overall_dist_weights(var_objective, dual_costs, edge_index_var_con):
    var_indices = edge_index_var_con[0]
    edge_objective = var_objective[var_indices]
    return dual_costs / (edge_objective + 1e-6)

def normalize_objective_batch(objective_batch, batch_indices):
    norm_batch = torch.sqrt(scatter_sum(torch.square(objective_batch), batch_indices))
    normalized_objective = objective_batch / norm_batch[batch_indices]
    return normalized_objective, norm_batch