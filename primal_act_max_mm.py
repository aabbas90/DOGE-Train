import torch
import numpy as np
from torch_scatter.scatter import scatter_add
torch.use_deterministic_algorithms(False)
from data.dataloader import get_ilp_gnn_loaders
from config_primal.defaults import get_cfg_defaults
import model.solver_utils as sol_utils

def transfer_batch_to_device(batch, device):
    batch.edge_index_var_con = batch.edge_index_var_con.to(device)
    batch.omega = torch.tensor([0.5], device = device)
    solvers, solver_state, per_bdd_sol, per_bdd_lb, dist_weights = sol_utils.init_solver_and_get_states(batch, device, 0, 1.0, batch.omega)
    all_mm_diff = sol_utils.compute_all_min_marginal_diff(solvers, solver_state)

    batch.dist_weights = dist_weights # Isotropic weights.

    var_degree = scatter_add(torch.ones((batch.num_edges), device=device), batch.edge_index_var_con[0])
    var_degree[torch.cumsum(batch.num_vars, 0) - 1] = 0 # Terminal nodes, not corresponding to any primal variable.
    var_net_perturbation = torch.zeros_like(var_degree)
    batch.var_lp_f = torch.stack((batch.objective.to(device), var_degree, var_net_perturbation), 1) # Obj, Deg, Net. Pert
    batch.objective = None

    con_degree = scatter_add(torch.ones((batch.num_edges), device=device), batch.edge_index_var_con[1])
    batch.con_lp_f = torch.stack((per_bdd_lb, batch.rhs_vector.to(device), batch.con_type.to(device), con_degree), 1) # BDD lb, rhs, con type, degree
    batch.rhs_vector = None
    batch.con_type = None

    # Edge LP features:
    batch.edge_rest_lp_f = torch.stack((per_bdd_sol, batch.con_coeff.to(device), all_mm_diff, all_mm_diff), 1)
    batch.solver_state = solver_state
    
    batch.solvers = solvers
    return batch

def get_valid_target_solution_edge(batch):
    valid_edge_mask = sol_utils.get_valid_edge_mask(batch)
    var_indices = batch.edge_index_var_con[0]
    valid_var_indices = var_indices[valid_edge_mask]
    gt_ilp_sol_var = torch.cat([torch.from_numpy(s).to(var_indices.device) for s in batch.gt_info['ilp_stats']['sol']], 0)
    gt_ilp_sol_edge = gt_ilp_sol_var[valid_var_indices]
    return gt_ilp_sol_edge, valid_edge_mask

def primal_loss(mm_pred, gt_ilp_sol_edge, valid_edge_mask, loss_margin):
    # Gather the solution w.r.t all valid edges
    mm_pred_valid = mm_pred[valid_edge_mask]

    # if gt_ilp_solution > 0 then mm_pred should be < -eps and if gt_ilp_solution == 0 then mm_pred should be > eps:
    return torch.sum(torch.relu(gt_ilp_sol_edge * (mm_pred_valid + loss_margin)) + torch.relu((gt_ilp_sol_edge - 1.0) * (mm_pred_valid - loss_margin)))

cfg = get_cfg_defaults()
combined_train_loader, test_loaders, test_datanames = get_ilp_gnn_loaders(cfg)
device = torch.device('cuda:0')
num_dual_iterations = 20
omega = torch.tensor(0.5)
batch = next(iter(combined_train_loader))
batch = transfer_batch_to_device(batch, device)
gt_ilp_sol_edge, valid_edge_mask = get_valid_target_solution_edge(batch)
pert = torch.zeros_like(batch.solver_state['lo_costs'])
lo_costs = batch.solver_state['lo_costs']
hi_costs = batch.solver_state['hi_costs']
def_mm = batch.solver_state['def_mm']
prev_loss = 0
for i in range(100):
    dual_cost_perturbation = pert.clone().detach()
    solver_state = {'lo_costs': lo_costs.clone().detach(), 'hi_costs': hi_costs.clone().detach(), 'def_mm': def_mm.clone().detach()}
    dual_cost_perturbation.requires_grad = True
    dual_cost_perturbation.retain_grad()
    new_lo = solver_state['lo_costs']  #+ torch.relu(-dual_cost_perturbation)
    new_hi = solver_state['hi_costs'] + dual_cost_perturbation  #+ torch.relu(dual_cost_perturbation)
    new_hi.retain_grad()
    new_solver_state = {'lo_costs': new_lo, 'hi_costs': new_hi, 'def_mm': def_mm.clone().detach()}
    solver_state_itr = sol_utils.dual_iterations(batch.solvers, new_solver_state, batch.dist_weights, num_dual_iterations, omega, 1e-6, 0)
    solver_state_itr['hi_costs'].retain_grad()
    solver_state_dd = sol_utils.distribute_delta(batch.solvers, solver_state_itr)
    solver_state_dd['hi_costs'].retain_grad()
    new_mm = sol_utils.compute_all_min_marginal_diff(batch.solvers, solver_state_dd)
    new_mm.retain_grad()
    # if i > 3:
    #     breakpoint()
    current_loss = primal_loss(new_mm, gt_ilp_sol_edge, valid_edge_mask, 1e-3)
    current_loss.backward()
    # if current_loss.item() == prev_loss:
    #     breakpoint()
    prev_loss = current_loss.item()
    print(f'{i}: {current_loss.item()}')
    pert = pert - 1e-2 * dual_cost_perturbation.grad
    print(f'dual_cost_perturbation.grad: {dual_cost_perturbation.grad.abs().max():.4f}, new_hi: {new_hi.grad.abs().max():.4f}, new_mm: {new_mm.grad.abs().max()}')

breakpoint()