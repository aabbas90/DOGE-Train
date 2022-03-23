import torch
import pickle
from torch_scatter import scatter_sum, scatter_softmax, scatter_mean, scatter_add, scatter_std
import bdd_cuda_torch
import numpy as np

def set_features(f_name, feature_names_to_use, feature_tensor, feature_vector):
    for (i, name) in enumerate(feature_names_to_use):
        if f_name == name:
            feature_tensor[:, i] = feature_vector
    return feature_tensor

def init_solver_and_get_states(batch, device, gt_solution_type, var_lp_f_names, con_lp_f_names, edge_lp_f_names, 
                            num_iterations = 0, improvement_slope = 1e-6, omega = 0.5, distribute_delta = False):
    batch.edge_index_var_con = batch.edge_index_var_con.to(device) 
    batch.batch_index_var = batch.objective_batch.to(device) # var batch assignment
    batch.batch_index_con = batch.rhs_vector_batch.to(device) # con batch assignment
    batch.batch_index_edge = batch.edge_index_var_con_batch.to(device) # edge batch assignment
    batch.num_cons = batch.num_cons.to(device)
    num_edges = batch.edge_index_var_con.shape[1]
    batch.omega = torch.tensor([omega], device = device)
    lo_costs_out = torch.empty((num_edges), device = device, dtype = torch.float32)
    hi_costs_out = torch.empty_like(lo_costs_out)
    def_mm_out = torch.empty_like(lo_costs_out)
    solvers = []
    initial_lbs = []
    layer_start = 0
    for b in range(len(batch.solver_data)):
        solver = pickle.loads(batch.solver_data[b])
        batch.solver_data[b] = None # Free-up the memory.
        # Run non learned solver for num_iterations e.g. to take it to states where optimization becomes difficult.
        solver.non_learned_iterations(omega, num_iterations, improvement_slope)
        if distribute_delta:
            solver.distribute_delta()
        initial_lbs.append(solver.lower_bound())
        solver.get_solver_costs(lo_costs_out[layer_start].data_ptr(), hi_costs_out[layer_start].data_ptr(), def_mm_out[layer_start].data_ptr())
        layer_start += solver.nr_layers()
        solvers.append(solver)

    assert(layer_start == lo_costs_out.shape[0])

    solver_state = {'lo_costs': lo_costs_out, 'hi_costs': hi_costs_out, 'def_mm': def_mm_out}
    per_bdd_lb = compute_per_bdd_lower_bound(solvers, solver_state)
    per_bdd_sol = compute_per_bdd_solution(solvers, solver_state)
    dist_weights = normalize_distribution_weights(torch.ones_like(lo_costs_out), batch.edge_index_var_con)
    valid_edge_mask_list = get_valid_edge_mask(batch.edge_index_var_con, solvers)
    if gt_solution_type is not None:
        gt_sol_var, gt_sol_edge = get_valid_target_solution_edge(solvers, batch.num_vars, batch.edge_index_var_con[0], batch.gt_info, valid_edge_mask_list, gt_solution_type)
    else:
        gt_sol_var = None
        gt_sol_edge = None
    valid_edge_mask = torch.cat(valid_edge_mask_list, 0)
    batch.valid_edge_mask = valid_edge_mask
    batch.gt_sol_edge = gt_sol_edge
    batch.gt_sol_var = gt_sol_var
    batch.initial_lbs = initial_lbs
    batch.solver_state = solver_state
    batch.solvers = solvers
    mm_diff = compute_all_min_marginal_diff(solvers, solver_state)
    try:
        assert(torch.all(torch.isfinite(solver_state['lo_costs'])))
        assert(torch.all(torch.isfinite(solver_state['hi_costs'])))
        assert(torch.all(torch.isfinite(solver_state['def_mm'])))
        assert(torch.all(torch.isfinite(mm_diff)))
    except:
        breakpoint()

    # Variable LP features:
    var_degree = scatter_add(torch.ones((batch.num_edges), device=device), batch.edge_index_var_con[0])
    var_degree[torch.cumsum(batch.num_vars, 0) - 1] = 0 # Terminal nodes, not corresponding to any primal variable.
    batch.var_lp_f = torch.zeros((torch.numel(batch.objective), len(var_lp_f_names)), device = device, dtype = torch.float32)
    batch.var_lp_f = set_features('obj', var_lp_f_names, batch.var_lp_f, batch.objective.to(device))
    batch.var_lp_f = set_features('deg', var_lp_f_names, batch.var_lp_f, var_degree)

    # Constraint LP features:
    con_degree = scatter_add(torch.ones((batch.num_edges), device=device), batch.edge_index_var_con[1])
    batch.con_lp_f = torch.zeros((torch.numel(con_degree), len(con_lp_f_names)), device = device, dtype = torch.float32)
    batch.con_lp_f = set_features('deg', con_lp_f_names, batch.con_lp_f, con_degree)
    batch.con_lp_f = set_features('rhs', con_lp_f_names, batch.con_lp_f, batch.rhs_vector.to(device))
    batch.con_lp_f = set_features('lb', con_lp_f_names, batch.con_lp_f, per_bdd_lb)
    batch.con_lp_f = set_features('con_type', con_lp_f_names, batch.con_lp_f, batch.con_type.to(device))
    
    # Edge LP features:
    batch.edge_rest_lp_f = torch.zeros((torch.numel(dist_weights), len(edge_lp_f_names)), device = device, dtype = torch.float32)
    batch.edge_rest_lp_f = set_features('sol', edge_lp_f_names, batch.edge_rest_lp_f, per_bdd_sol)
    batch.edge_rest_lp_f = set_features('dist_weights', edge_lp_f_names, batch.edge_rest_lp_f, dist_weights)
    batch.edge_rest_lp_f = set_features('mm_diff', edge_lp_f_names, batch.edge_rest_lp_f, mm_diff)
    batch.edge_rest_lp_f = set_features('coeff', edge_lp_f_names, batch.edge_rest_lp_f, batch.con_coeff.to(device))
    return batch, dist_weights

def normalize_costs_var(var_costs, lb_per_bdd, mm_diff, num_bdds_per_inst, batch_index_var, batch_index_con, batch_index_edge):
    m = scatter_mean(var_costs, batch_index_var)
    std = scatter_std(var_costs, batch_index_var) + 1e-8
    var_costs = (var_costs - m[batch_index_var]) / (std[batch_index_var])
    mean_per_bdd = m / num_bdds_per_inst
    lb_per_bdd = (lb_per_bdd - mean_per_bdd[batch_index_con]) / (std[batch_index_con])
    mm_diff = mm_diff / (std[batch_index_edge])
    return var_costs, lb_per_bdd, mm_diff, m, std

def non_learned_iterations(solvers, solver_state, num_iterations = 0, improvement_slope = 1e-6, omega = 0.5):
    layer_start = 0
    for (b, solver) in enumerate(solvers):  
        solver.set_solver_costs(solver_state['lo_costs'][layer_start].data_ptr(), 
                                solver_state['hi_costs'][layer_start].data_ptr(), 
                                solver_state['def_mm'][layer_start].data_ptr())
        solver.non_learned_iterations(omega, num_iterations, improvement_slope)
        solver.get_solver_costs(solver_state['lo_costs'][layer_start].data_ptr(), 
                                solver_state['hi_costs'][layer_start].data_ptr(), 
                                solver_state['def_mm'][layer_start].data_ptr())
        layer_start += solver.nr_layers()
    return solvers, solver_state

def non_learned_updates(batch, num_iterations = 0, improvement_slope = 1e-6, omega = 0.5):
    batch.solvers, batch.solver_state = non_learned_iterations(batch.solvers, batch.solver_state, num_iterations, improvement_slope, omega)
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

def get_valid_edge_mask(edge_index_var_con, solvers):
    masks_list = [] #torch.ones(batch.num_edges, dtype=torch.bool)
    for (b, solver) in enumerate(solvers):
        terminal_indices = torch.empty((solver.nr_bdds()), device = edge_index_var_con.device, dtype = torch.int32)
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
    num_caches = min(int(grad_dual_itr_max_itr / 3), 25)
    lo_costs, hi_costs, def_mm = bdd_cuda_torch.DualIterations.apply(solvers, solver_state['lo_costs'], solver_state['hi_costs'], solver_state['def_mm'], 
                                                                    dist_weights, num_iterations, omega, grad_dual_itr_max_itr, improvement_slope, num_caches)
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

def perturb_primal_costs(lo_costs, hi_costs, lo_pert, hi_pert, dist_weights, edge_index_var_con):
    var_indices = edge_index_var_con[0]
    lo_costs = lo_costs + lo_pert[var_indices] * dist_weights
    hi_costs = hi_costs + hi_pert[var_indices] * dist_weights
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

def primal_rounding_non_learned(num_rounds, solvers, norm_solver_state, obj_multiplier, obj_offset, num_itr_lb, improvement_slope_lb, omega, edge_index_var_con, dist_weights, init_delta = 1.0, delta_growth_rate = 1.2):
    assert len(solvers) == 1, 'batch size 1 supported for now.'
    hi_costs = norm_solver_state['hi_costs'] / obj_multiplier[0].item() + obj_offset[0].item()
    lo_costs = norm_solver_state['lo_costs'] / obj_multiplier[0].item() + obj_offset[0].item()
    solver_state = {'hi_costs': hi_costs, 'lo_costs': lo_costs, 'def_mm': torch.zeros_like(lo_costs)}
    logs = []
    for round_index in range(num_rounds):
        var_indices = edge_index_var_con[0]
        current_delta = min(init_delta * np.power(delta_growth_rate, round_index), 1e6)

        mm_diff = compute_all_min_marginal_diff(solvers, solver_state)
        logs.append({'r' : round_index, 'all_mm_diff': mm_diff.detach()})

        hi_assignments = mm_diff < -1e-6
        lo_assignments = mm_diff > 1e-6

        undecided_assigments = torch.logical_and(~hi_assignments, ~lo_assignments)
        var_hi = scatter_mean(hi_assignments.to(torch.float32), var_indices) >= 1.0 - 1e-6
        var_lo = scatter_mean(lo_assignments.to(torch.float32), var_indices) >= 1.0 - 1e-6
        var_lo[-1] = True # terminal node.
        if (var_hi + var_lo).min() >= 1.0 - 1e-6: # Solution found
            return mm_diff, var_hi, logs
        
        var_undecided = scatter_mean(undecided_assigments.to(torch.float32), var_indices) >= 1.0 - 1e-6
        var_inconsistent = torch.logical_and(torch.logical_and(~var_hi, ~var_lo), ~var_undecided)

        rand_perturbation = 2.0 * current_delta * torch.rand(var_undecided.shape, dtype = torch.float32, device = mm_diff.device) - current_delta
        var_undecided_pert = var_undecided * rand_perturbation

        mm_diff_sum_sign = torch.sign(scatter_sum(mm_diff, var_indices))
        var_inconsistent_pert = var_inconsistent * mm_diff_sum_sign * torch.abs(rand_perturbation)
        var_cost_pert = -(var_hi * current_delta) + (var_lo * current_delta) + (var_undecided_pert) + (var_inconsistent_pert)
        var_cost_pert[-1] = 0 # dummy primal variable for terminal nodes.
        solver_state['lo_costs'], solver_state['hi_costs'] = perturb_primal_costs(solver_state['lo_costs'], 
                                                                                solver_state['hi_costs'], 
                                                                                torch.relu(-var_cost_pert),
                                                                                torch.relu(var_cost_pert),
                                                                                dist_weights,
                                                                                edge_index_var_con)
        solvers, solver_state = non_learned_iterations(solvers, solver_state, num_itr_lb, improvement_slope_lb, omega)
        solver_state = distribute_delta(solvers, solver_state)
        # print(f'one min-marg diff.: {var_hi.sum():.0f}, '
        #     f'zero min-marg diff.: {var_lo.sum():.0f}, '
        #     f'equal min-marg diff.: {var_undecided.sum():.0f}, '
        #     f'inconsistent min-marg diff.: {var_inconsistent.sum():.0f}')
    return mm_diff, None, logs

