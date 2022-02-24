import torch
import pickle
import BDD.bdd_cuda_learned_mma_py as bdd_solver
from torch_scatter import scatter_sum, scatter_mean
import bdd_cuda_torch
import time 

def init_solver_and_get_states(batch, device, num_iterations = 0, improvement_slope = 1e-6, omega = 0.5):
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
    try:
        assert(torch.all(torch.isfinite(def_mm_out)))
    except:
        breakpoint()

    solver_state = {'lo_costs': lo_costs_out, 'hi_costs': hi_costs_out, 'def_mm': def_mm_out}
    per_bdd_lb = compute_per_bdd_lower_bound(solvers, solver_state)
    per_bdd_sol = compute_per_bdd_solution(solvers, solver_state)
    dist_weights = normalize_distribution_weights(torch.ones_like(lo_costs_out), batch.edge_index_var_con)
    return solvers, solver_state, per_bdd_sol, per_bdd_lb, dist_weights

def get_valid_edge_mask(batch):
    mask = torch.ones(batch.num_edges, dtype=torch.bool)
    layer_start = 0
    for (b, solver) in enumerate(batch.solvers):
        terminal_indices = torch.empty((solver.nr_bdds()), device = batch.edge_index_var_con.device, dtype = torch.int32)
        solver.terminal_layer_indices(terminal_indices.data_ptr())
        mask[terminal_indices.to(torch.long) + layer_start] = False
        layer_start += solver.nr_layers()
    return mask

def compute_per_bdd_lower_bound(solvers, solver_state):
    return bdd_cuda_torch.ComputeLowerBoundperBDD.apply(solvers, solver_state['lo_costs'], solver_state['hi_costs'])

def compute_per_bdd_solution(solvers, solver_state):
    return bdd_cuda_torch.ComputePerBDDSolutions(solvers, solver_state['lo_costs'], solver_state['hi_costs'])

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
    return bdd_cuda_torch.ComputeAllMinMarginalsDiff.apply(solvers, solver_state['lo_costs'], solver_state['hi_costs'])

def normalize_distribution_weights(dist_weights, edge_index_var_con):
# Normalize distribution weights so that they sum upto 1 for each variable.
    var_indices = edge_index_var_con[0, :]
    dist_weights_sum = scatter_sum(dist_weights, var_indices)[var_indices]
    return dist_weights / dist_weights_sum

def perturb_primal_costs(solvers, solver_state, primal_perturbation):
    assert(solver_state.shape[1] == 2)
    primal_pert_lo = torch.nn.ReLU(-primal_perturbation)
    primal_pert_hi = torch.nn.ReLU(primal_perturbation)
    lo_costs, hi_costs = bdd_cuda_torch.PerturbPrimalCosts.apply(solvers, primal_pert_lo, primal_pert_hi, solver_state['lo_costs'], solver_state['hi_costs'])
    solver_state['lo_costs'] = lo_costs
    solver_state['hi_costs'] = hi_costs
    return solver_state

def check_decode_primal_solution(batch, mm_pred, orig_obj_vector, gt_ilp_sol_var):
    var_indices = batch.edge_index_var_con[0]
    mm_pred_sign = torch.sign(mm_pred)
    variable_mean_sign = scatter_mean(mm_pred_sign, var_indices)
    assignment_lo = variable_mean_sign >= 0.9
    assignment_hi = variable_mean_sign <= -0.9
    vars_agree = torch.logical_or(assignment_hi, assignment_lo)
    solutions = []
    solution_objectives = []
    gt_sol_objectives = []
    prev_var_start = 0
    fraction_disagreements = []
    for (b, solver) in enumerate(batch.solvers):
        var_end = batch.num_vars[b] + prev_var_start - 1
        is_decodable = torch.all(vars_agree[prev_var_start:var_end])
        if is_decodable:
            current_sol = torch.zeros((batch.num_vars[b]), device=mm_pred.device)
            current_sol[assignment_hi[prev_var_start:var_end + 1]] = 1.0
            current_sol[-1] = 0 # Terminal node.
            solutions.append(current_sol)

            current_sol_cost = torch.sum(orig_obj_vector[prev_var_start:var_end + 1] * current_sol)
            solution_objectives.append(current_sol_cost)
            gt_sol_cost = torch.sum(orig_obj_vector[prev_var_start:var_end + 1] * gt_ilp_sol_var[prev_var_start:var_end + 1])
            gt_sol_objectives.append(gt_sol_cost)
            fraction_disagreements.append(torch.tensor([0.0]))
        else:
            fraction_disagreements.append(torch.logical_not(vars_agree[prev_var_start:var_end]).sum().item() / batch.num_vars[b])
        prev_var_start = var_end + 1 # To account for terminal node.
    return solutions, solution_objectives, gt_sol_objectives, fraction_disagreements