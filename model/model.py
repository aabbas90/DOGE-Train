import torch
import torch.nn as nn
from torch_scatter import scatter_sum, scatter_mean, scatter_max
from torch_geometric.nn import TransformerConv, GATv2Conv, LayerNorm
import model.solver_utils as sol_utils

# https://github.com/rusty1s/pytorch_geometric/issues/1210 might be useful
# https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html?highlight=meta%20layer#torch_geometric.nn.meta.MetaLayer
class EdgeUpdater(torch.nn.Module):
    # Receives learned var, con, edge features and fixed edge features to predict new learned edge features.
    def __init__(self, num_input_edge_channels, num_hidden_channels, num_output_edge_channels, num_var_channels, num_con_channels, num_hidden_layers = 0):
        super(EdgeUpdater, self).__init__()
        # num_input_channels = num_var_channels + num_con_channels + num_input_edge_channels
        self.var_compress_mlp = [nn.Linear(num_var_channels, num_hidden_channels), nn.ReLU(True)]
        for l in range(num_hidden_layers + 1):
            self.var_compress_mlp.append(nn.Linear(num_hidden_channels, num_hidden_channels))
            self.var_compress_mlp.append(nn.ReLU(True))       
        self.var_compress_mlp = nn.Sequential(*self.var_compress_mlp)

        self.con_compress_mlp = [nn.Linear(num_con_channels, num_hidden_channels), nn.ReLU(True)]
        for l in range(num_hidden_layers + 1):
            self.con_compress_mlp.append(nn.Linear(num_hidden_channels, num_hidden_channels))
            self.con_compress_mlp.append(nn.ReLU(True))
        self.con_compress_mlp = nn.Sequential(*self.con_compress_mlp)

        self.edge_mlp = [nn.Linear(num_input_edge_channels + num_hidden_channels + num_hidden_channels, num_hidden_channels), nn.ReLU(True)]
        for l in range(num_hidden_layers):
            self.edge_mlp.append(nn.Linear(num_hidden_channels, num_hidden_channels))
            self.edge_mlp.append(nn.ReLU(True))
        self.edge_mlp.append(nn.Linear(num_hidden_channels, num_output_edge_channels))
        self.edge_mlp = nn.Sequential(*self.edge_mlp)
        # m = torch.normal(0.0, 1e-6, (num_learned_edge_channels, num_input_channels)) 
        # m[:, mm_1_index] = 1.0
        # m[:, mm_0_index] = -1.0
        # self.edge_mlp[0].weight.data = m

    def forward(self, var_f: torch.Tensor, con_f: torch.Tensor, combined_edge_f: torch.Tensor, edge_index_var_con: torch.Tensor):
        out = torch.cat([combined_edge_f, 
                            self.var_compress_mlp(var_f)[edge_index_var_con[0], :], 
                            self.con_compress_mlp(con_f)[edge_index_var_con[1], :]
                        ], 1)
        return self.edge_mlp(out)

def compute_normalized_solver_costs_for_dual(solver_state, batch_index_edge, norm_type = 'l2'):
    net_cost = solver_state['hi_costs'] - solver_state['lo_costs'] + solver_state['def_mm']
    if norm_type == 'l2':
        norm = scatter_sum(torch.square(net_cost), batch_index_edge)[batch_index_edge]
    elif norm_type == 'inf':
        norm = scatter_max(torch.abs(net_cost), batch_index_edge)[0][batch_index_edge]
    else:
        assert False, f'norm: {norm_type} unknown'
    norm_cost = net_cost / norm
    norm_def_mm = solver_state['def_mm'] / (norm + 1e-9)
    norm_state = {'norm_cost': norm_cost, 'norm_def_mm': norm_def_mm}
    return norm_state

class FeatureExtractorLayer(torch.nn.Module):
    def __init__(self, 
                num_var_lp_f, in_var_dim, out_var_dim, 
                num_con_lp_f, in_con_dim, out_con_dim,
                num_edge_lp_f_with_ss, in_edge_dim, out_edge_dim,
                use_layer_norm, use_def_mm = True, use_solver_costs = True,
                use_net_solver_costs = False,
                num_hidden_layers_edge = 0):
        super(FeatureExtractorLayer, self).__init__()

        self.use_def_mm = use_def_mm
        self.use_solver_costs = use_solver_costs
        self.con_updater = TransformerConv((num_var_lp_f + in_var_dim, num_con_lp_f + in_con_dim), 
                                            out_con_dim, 
                                            edge_dim = num_edge_lp_f_with_ss + in_edge_dim, 
                                            flow="source_to_target", aggr = 'mean')
        self.var_updater = TransformerConv((out_con_dim + num_con_lp_f, num_var_lp_f + in_var_dim), 
                                            out_var_dim,
                                            edge_dim = num_edge_lp_f_with_ss + in_edge_dim, 
                                            flow="target_to_source", aggr = 'mean')

        self.edge_updater = EdgeUpdater(num_edge_lp_f_with_ss + in_edge_dim, out_edge_dim, out_edge_dim, out_var_dim + num_var_lp_f, out_con_dim + num_con_lp_f, num_hidden_layers_edge)
        self.use_net_solver_costs = use_net_solver_costs

        if use_layer_norm:
            self.var_norm = LayerNorm(out_var_dim, affine = True)
            self.con_norm = LayerNorm(out_con_dim, affine = True)
            self.edge_norm = LayerNorm(out_edge_dim, affine = True)
        else:
            self.var_norm = torch.nn.Identity()
            self.con_norm = torch.nn.Identity()
            self.edge_norm = torch.nn.Identity()

    def combine_features(self, f1, f2):
        return torch.cat((f1, f2), 1)

    def forward(self, var_learned_f, var_lp_f, con_learned_f, con_lp_f, edge_learned_f, solver_state, edge_lp_f_wo_ss, edge_index_var_con, batch_index_var = None, batch_index_con = None, batch_index_edge = None):
        # 0. Combine learned and fixed features:
        var_comb_f = self.combine_features(var_learned_f, var_lp_f)
        con_comb_f = self.combine_features(con_learned_f, con_lp_f)
        if self.use_solver_costs:
            if self.use_net_solver_costs:
                if self.use_def_mm:
                    edge_comb_f = torch.cat((edge_learned_f, solver_state['norm_cost'].unsqueeze(1), solver_state['norm_def_mm'].unsqueeze(1), edge_lp_f_wo_ss), 1)
                else:
                    edge_comb_f = torch.cat((edge_learned_f, solver_state['norm_cost'].unsqueeze(1), edge_lp_f_wo_ss), 1)
            else:
                if self.use_def_mm:
                    edge_comb_f = torch.cat((edge_learned_f, solver_state['lo_costs'].unsqueeze(1), solver_state['hi_costs'].unsqueeze(1), solver_state['def_mm'].unsqueeze(1), edge_lp_f_wo_ss), 1)
                else:
                    edge_comb_f = torch.cat((edge_learned_f, solver_state['lo_costs'].unsqueeze(1), solver_state['hi_costs'].unsqueeze(1), edge_lp_f_wo_ss), 1)
        else:
            assert(not self.use_def_mm)        
            edge_comb_f = torch.cat((edge_learned_f, edge_lp_f_wo_ss), 1)

        # 1. Update constraint features (var, constraint, edge -> constraint_new):
        con_learned_f = torch.relu(self.con_norm(self.con_updater((var_comb_f, con_comb_f), edge_index_var_con, edge_comb_f), batch_index_con))
        con_comb_f = self.combine_features(con_learned_f, con_lp_f)

        # 2. Update variable features (var, constraint_new, edge -> var):
        var_learned_f = torch.relu(self.var_norm(self.var_updater((con_comb_f, var_comb_f), edge_index_var_con, edge_comb_f), batch_index_var))
        var_comb_f = self.combine_features(var_learned_f, var_lp_f)

        # 3. Update edges:
        edge_learned_f = torch.relu(self.edge_norm(self.edge_updater(var_comb_f, con_comb_f, edge_comb_f, edge_index_var_con), batch_index_edge))

        return var_learned_f, con_learned_f, edge_learned_f

def get_edge_mask_without_terminal_nodes(edge_index_var_con, var_degree):
    var_indices = edge_index_var_con[0]
    non_terminal_vars = var_degree != 0
    non_terminal_edges_mask = non_terminal_vars[var_indices]
    return non_terminal_edges_mask

class FeatureExtractor(torch.nn.Module):
    def __init__(self, num_var_lp_f, out_var_dim, num_con_lp_f, out_con_dim, num_edge_lp_f, out_edge_dim, depth, 
                use_layer_norm = False, use_def_mm = True, num_hidden_layers_edge = 0, use_net_solver_costs = False,
                takes_learned_features = False):
        super(FeatureExtractor, self).__init__()
        self.num_var_lp_f = num_var_lp_f
        self.num_con_lp_f = num_con_lp_f
        num_edge_lp_f_with_ss = num_edge_lp_f + 2 - int(use_net_solver_costs) + int(use_def_mm) 
        self.use_def_mm = use_def_mm
        self.takes_learned_features = takes_learned_features
        if takes_learned_features:
            self.in_var_dim = out_var_dim
            self.in_con_dim = out_con_dim
            self.in_edge_dim = out_edge_dim
        else:
            self.in_var_dim = 0
            self.in_con_dim = 0
            self.in_edge_dim = 0

        layers = [
            FeatureExtractorLayer(num_var_lp_f, self.in_var_dim, out_var_dim,
                                num_con_lp_f, self.in_con_dim, out_con_dim,
                                num_edge_lp_f_with_ss, self.in_edge_dim, out_edge_dim,
                                use_layer_norm, use_def_mm, 
                                num_hidden_layers_edge = num_hidden_layers_edge,
                                use_net_solver_costs = use_net_solver_costs)
        ]
        for l in range(depth - 1):
            layers.append(
                FeatureExtractorLayer(num_var_lp_f, out_var_dim, out_var_dim,
                                    num_con_lp_f, out_con_dim, out_con_dim,
                                    num_edge_lp_f_with_ss, out_edge_dim, out_edge_dim,
                                    use_layer_norm, use_def_mm,
                                    num_hidden_layers_edge = num_hidden_layers_edge,
                                    use_net_solver_costs = use_net_solver_costs)
            )
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, var_lp_f, con_lp_f, solver_state, edge_lp_f_wo_ss, edge_index_var_con, 
                var_learned_f = None, con_learned_f = None, edge_learned_f = None,
                batch_index_var = None, batch_index_con = None, batch_index_edge = None):
        if self.takes_learned_features:
            assert var_learned_f is not None
            assert con_learned_f is not None
            assert edge_learned_f is not None
        else:
            var_learned_f = torch.zeros((var_lp_f.shape[0], 0), device = var_lp_f.device)
            con_learned_f = torch.zeros((con_lp_f.shape[0], 0), device = con_lp_f.device)
            edge_learned_f = torch.zeros((edge_lp_f_wo_ss.shape[0], 0), device = edge_lp_f_wo_ss.device)

        for l in range(len(self.layers)):
            var_learned_f, con_learned_f, edge_learned_f = self.layers[l](
                var_learned_f, var_lp_f, con_learned_f, con_lp_f, edge_learned_f, solver_state, edge_lp_f_wo_ss,
                edge_index_var_con, batch_index_var, batch_index_con, batch_index_edge)
            
        return var_learned_f, con_learned_f, edge_learned_f

class PrimalPerturbationBlock(torch.nn.Module):
    def __init__(self, var_lp_f_names, con_lp_f_names, edge_lp_f_names, depth, var_dim, con_dim, edge_dim, min_perturbation, is_first_block = True,
                use_layer_norm = False, use_lstm_var = False, num_hidden_layers_edge = 0):
        super(PrimalPerturbationBlock, self).__init__()
        self.min_perturbation = min_perturbation
        self.var_lp_f_names = var_lp_f_names
        self.con_lp_f_names = con_lp_f_names
        self.edge_lp_f_names = edge_lp_f_names
        self.num_var_lp_f = len(var_lp_f_names)
        self.num_con_lp_f = len(con_lp_f_names)
        self.num_edge_lp_f_with_ss = len(edge_lp_f_names)
        self.is_first_block = is_first_block
        if is_first_block:
            assert not use_lstm_var
            self.feature_refinement = FeatureExtractor(self.num_var_lp_f, var_dim, self.num_con_lp_f, con_dim, self.num_edge_lp_f_with_ss, edge_dim, 
                                                    depth, use_layer_norm, use_def_mm = False, num_hidden_layers_edge = num_hidden_layers_edge,
                                                    use_net_solver_costs = True, takes_learned_features = False)
        else:
            self.feature_refinement = FeatureExtractor(self.num_var_lp_f, var_dim, self.num_con_lp_f, con_dim, self.num_edge_lp_f_with_ss, edge_dim, 
                                                    depth, use_layer_norm, use_def_mm = False, num_hidden_layers_edge = num_hidden_layers_edge,
                                                    use_net_solver_costs = True, takes_learned_features = True)

        self.var_lstm = None
        in_var_dim = var_dim
        if use_lstm_var:
            self.var_lstm = nn.LSTMCell(var_dim, var_dim)
            in_var_dim += var_dim
        self.var_pert_predictor = nn.Sequential(nn.Linear(in_var_dim, 2 * var_dim), nn.ReLU(True), 
                                                nn.Linear(2 * var_dim, var_dim), nn.ReLU(True), 
                                                nn.Linear(var_dim, 1))
        
    def forward(self, solvers, var_lp_f, con_lp_f, orig_solver_state, norm_orig_solver_state, edge_lp_f_wo_ss, 
                var_learned_f, con_learned_f, edge_learned_f, 
                dist_weights, omega, edge_index_var_con,
                num_dual_iterations, grad_dual_itr_max_itr, dual_improvement_slope,
                batch_index_var, batch_index_con, batch_index_edge, var_hidden_states_lstm = None):

        var_learned_f, con_learned_f, edge_learned_f = self.feature_refinement(var_lp_f, con_lp_f, norm_orig_solver_state, edge_lp_f_wo_ss, edge_index_var_con,
                                                                            var_learned_f, con_learned_f, edge_learned_f, 
                                                                            batch_index_var, batch_index_con, batch_index_edge)
        if self.var_lstm is not None:
            assert(var_hidden_states_lstm is not None)
            lstm_h, lstm_c = self.var_lstm(var_learned_f, (var_hidden_states_lstm['h'], var_hidden_states_lstm['c']))
            primal_perturbation_var = self.var_pert_predictor(torch.cat((lstm_h, var_learned_f), 1)).squeeze()
            var_hidden_states_lstm['h'] = lstm_h
            var_hidden_states_lstm['c'] = lstm_c
        else:
            primal_perturbation_var = self.var_pert_predictor(var_learned_f).squeeze()

        primal_perturbation_var_lo = -primal_perturbation_var + self.min_perturbation * 0.5
        primal_perturbation_var_hi = primal_perturbation_var + self.min_perturbation * 0.5

        p_lo_cost_var = torch.relu(primal_perturbation_var_lo)
        p_hi_cost_var = torch.relu(primal_perturbation_var_hi)
        
        p_lo_cost = p_lo_cost_var[edge_index_var_con[0]] * dist_weights
        p_hi_cost = p_hi_cost_var[edge_index_var_con[0]] * dist_weights

        try:
            assert(torch.all(torch.isfinite(primal_perturbation_var)))
        except:
            breakpoint()

        pert_solver_state = {
                            'lo_costs': orig_solver_state['lo_costs'] + p_lo_cost, 
                            'hi_costs': orig_solver_state['hi_costs'] + p_hi_cost,
                            'def_mm': orig_solver_state['def_mm']
                            }
        
        # Dual iterations
        pert_solver_state = sol_utils.dual_iterations(solvers, pert_solver_state, dist_weights, num_dual_iterations, omega, dual_improvement_slope, grad_dual_itr_max_itr)
        pert_solver_state = sol_utils.distribute_delta(solvers, pert_solver_state)

        # Normalize perturbed costs to send to GNN in next round:
        mm_diff = sol_utils.compute_all_min_marginal_diff(solvers, pert_solver_state)
        per_bdd_lb = sol_utils.compute_per_bdd_lower_bound(solvers, pert_solver_state) # Update perturbed lower bound.
        net_var_cost, _, mm_diff, per_bdd_lb, constant_per_bdd, norm = sol_utils.compute_normalized_solver_costs_for_primal(pert_solver_state, mm_diff, per_bdd_lb, 
                                                                edge_index_var_con, batch_index_var, batch_index_con, batch_index_edge)

        edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('prev_mm_diff')] = mm_diff
        con_lp_f[:, self.con_lp_f_names.index('prev_lb')] = per_bdd_lb
        var_lp_f[:, self.var_lp_f_names.index('prev_obj')] = net_var_cost
        con_lp_f[:, self.con_lp_f_names.index('prev_constant')] = constant_per_bdd
        
        # Update per BDD solution:
        with torch.no_grad(): #TODO: use black-box backprop?
            edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('prev_sol')] = sol_utils.compute_per_bdd_solution(solvers, pert_solver_state)

        if self.is_first_block:
            return var_learned_f, con_learned_f, edge_learned_f, var_lp_f, con_lp_f, edge_lp_f_wo_ss, norm
        else:
            return var_lp_f, con_lp_f, edge_lp_f_wo_ss, var_hidden_states_lstm, norm

class DualDistWeightsBlock(torch.nn.Module):
    def __init__(self, var_lp_f_names, con_lp_f_names, edge_lp_f_names, depth, var_dim, con_dim, edge_dim, 
                use_layer_norm = False, predict_omega = False, num_hidden_layers_edge = 0, use_net_solver_costs = False):
        super(DualDistWeightsBlock, self).__init__()
        self.var_lp_f_names = var_lp_f_names
        self.con_lp_f_names = con_lp_f_names
        self.edge_lp_f_names = edge_lp_f_names
        self.num_var_lp_f = len(var_lp_f_names)
        self.num_con_lp_f = len(con_lp_f_names)
        self.num_edge_lp_f_with_ss = len(edge_lp_f_names) + 3 - int(use_net_solver_costs)
        self.use_net_solver_costs = use_net_solver_costs
        self.feature_extractor = FeatureExtractor(self.num_var_lp_f, var_dim, self.num_con_lp_f, con_dim, len(edge_lp_f_names), edge_dim, 
                                                depth, use_layer_norm, use_def_mm = True, num_hidden_layers_edge = num_hidden_layers_edge,
                                                use_net_solver_costs = use_net_solver_costs)
        self.predict_omega = predict_omega
        num_outputs = 1 + int(predict_omega)
        self.dist_weights_predictor = EdgeUpdater(self.num_edge_lp_f_with_ss + edge_dim, edge_dim, num_outputs, 
                                            self.num_var_lp_f + var_dim, 
                                            self.num_con_lp_f + con_dim,
                                            num_hidden_layers_edge)

    def forward(self, solvers, var_lp_f, con_lp_f, 
                solver_state, edge_lp_f_wo_ss, 
                omega, edge_index_var_con,
                num_dual_iterations, grad_dual_itr_max_itr, dual_improvement_slope, valid_edge_mask, 
                batch_index_var, batch_index_con, batch_index_edge,
                logger, filepaths, step):
        
        if self.use_net_solver_costs:
            net_solver_state = compute_normalized_solver_costs_for_dual(solver_state, batch_index_edge, 'inf')
            var_learned_f, con_learned_f, edge_learned_f = self.feature_extractor(var_lp_f, con_lp_f, net_solver_state, edge_lp_f_wo_ss, edge_index_var_con, batch_index_var, batch_index_con, batch_index_edge)
        else:
            var_learned_f, con_learned_f, edge_learned_f = self.feature_extractor(var_lp_f, con_lp_f, solver_state, edge_lp_f_wo_ss, edge_index_var_con, batch_index_var, batch_index_con, batch_index_edge)

        if not self.use_net_solver_costs:
            predictions = self.dist_weights_predictor(
                torch.cat((var_learned_f, var_lp_f), 1), torch.cat((con_learned_f, con_lp_f), 1),
                torch.cat((edge_learned_f, solver_state['lo_costs'].unsqueeze(1), solver_state['hi_costs'].unsqueeze(1), solver_state['def_mm'].unsqueeze(1), edge_lp_f_wo_ss), 1), 
                edge_index_var_con)
        else:
            predictions = self.dist_weights_predictor(
                torch.cat((var_learned_f, var_lp_f), 1), torch.cat((con_learned_f, con_lp_f), 1),
                torch.cat((edge_learned_f, net_solver_state['norm_cost'].unsqueeze(1), net_solver_state['norm_def_mm'].unsqueeze(1), edge_lp_f_wo_ss), 1), 
                edge_index_var_con)

        dist_weights = predictions[:, 0]
        dist_weights = sol_utils.normalize_distribution_weights_softmax(dist_weights, edge_index_var_con)

        try:
            assert(torch.all(torch.isfinite(dist_weights)))
        except:
            breakpoint()

        if self.predict_omega:
            omega_vec = torch.sigmoid(predictions[:, 1])
            # Dual iterations
            solver_state = sol_utils.dual_iterations(solvers, solver_state, dist_weights, num_dual_iterations, omega_vec, dual_improvement_slope, grad_dual_itr_max_itr, logger, filepaths, step)
        else:
            omega_vec = None
            # Dual iterations
            solver_state = sol_utils.dual_iterations(solvers, solver_state, dist_weights, num_dual_iterations, omega, dual_improvement_slope, grad_dual_itr_max_itr)

        if 'prev_dist_weights' in self.edge_lp_f_names:
            edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('prev_dist_weights')] = dist_weights.clone().detach()
        elif 'dist_weights' in self.edge_lp_f_names:
            edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('dist_weights')] = dist_weights.clone().detach()
        if 'omega' in self.edge_lp_f_names:
            edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('omega')] = omega_vec.clone().detach()

        with torch.no_grad():
            if 'prev_sol' in self.edge_lp_f_names:
                edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('prev_sol')] = edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('sol')].clone()
            edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('sol')] = sol_utils.compute_per_bdd_solution(solvers, solver_state)
            if 'prev_lb' in self.con_lp_f_names:
                con_lp_f[:, self.con_lp_f_names.index('prev_lb')] = con_lp_f[:, self.con_lp_f_names.index('lb')].clone()
            if 'grad_dist_weights' in self.edge_lp_f_names:
                dist_weights_grad, omega_vec_grad = sol_utils.populate_grad_features_dual(solvers, solver_state, dist_weights, omega_vec, 20, batch_index_con)
                edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('grad_dist_weights')] = dist_weights_grad
                edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('grad_omega')] = omega_vec_grad

        con_lp_f[:, self.con_lp_f_names.index('lb')] = sol_utils.compute_per_bdd_lower_bound(solvers, solver_state)
       
        return solver_state, var_lp_f, con_lp_f, edge_lp_f_wo_ss, dist_weights, omega_vec

def run_full_coord_iterations(solvers, lo_costs, hi_costs, def_mm, new_mm, damping, dist_weights, var_indices, num_dual_iterations):
    solver_state = {'lo_costs': lo_costs, 'hi_costs': hi_costs, 'def_mm': def_mm}
    for dual_itr in range(num_dual_iterations):
        mm_to_subtract = damping * new_mm

        mm_to_subtract_lo = torch.relu(-mm_to_subtract)
        mm_to_subtract_hi = torch.relu(mm_to_subtract)

        # Update costs:
        solver_state['lo_costs'] = solver_state['lo_costs'] + scatter_sum(mm_to_subtract_lo, var_indices)[var_indices] * dist_weights - mm_to_subtract_lo
        solver_state['hi_costs'] = solver_state['hi_costs'] + scatter_sum(mm_to_subtract_hi, var_indices)[var_indices] * dist_weights - mm_to_subtract_hi

        #TODO Put a NN here?

        # Compute new min-marginal differences:
        new_mm = sol_utils.compute_all_min_marginal_diff(solvers, solver_state)
        new_lb = sol_utils.compute_per_bdd_lower_bound(solvers, solver_state)
        # print(f'new_mm: [{new_mm.min():.3f}, {new_mm.max():.3f}]')
        # print(f'new_lb: [{new_lb.min():.3f}, {new_lb.max():.3f}]')

    return solvers, solver_state, new_mm, new_lb
 
class DualFullCoordinateAscent(torch.nn.Module):
    def __init__(self, var_lp_f_names, con_lp_f_names, edge_lp_f_names, depth, var_dim, con_dim, edge_dim, use_layer_norm = False, skip_connections = False, num_hidden_layers_edge = 0):
        super(DualFullCoordinateAscent, self).__init__()
        self.var_lp_f_names = var_lp_f_names
        self.con_lp_f_names = con_lp_f_names
        self.edge_lp_f_names = edge_lp_f_names
        self.num_var_lp_f = len(var_lp_f_names)
        self.num_con_lp_f = len(con_lp_f_names)
        self.num_edge_lp_f_with_ss = len(edge_lp_f_names) + 2
        self.feature_refinement = []
        for d in range(depth):
            self.feature_refinement.append(FeatureExtractorLayer(self.num_var_lp_f, var_dim, var_dim,
                                                                self.num_con_lp_f, con_dim, con_dim,   
                                                                self.num_edge_lp_f_with_ss, edge_dim, edge_dim,
                                                                use_layer_norm, skip_connections, use_def_mm=False,
                                                                num_hidden_layers_edge = num_hidden_layers_edge))

        self.feature_refinement = torch.nn.ModuleList(self.feature_refinement)
        self.predictor = EdgeUpdater(self.num_edge_lp_f_with_ss + edge_dim, edge_dim, 2, self.num_var_lp_f + var_dim, self.num_con_lp_f + con_dim, num_hidden_layers_edge = num_hidden_layers_edge)
    
    def forward(self, solvers, var_lp_f, con_lp_f, 
                solver_state, edge_lp_f_wo_ss, 
                var_learned_f, con_learned_f, edge_learned_f, 
                edge_index_var_con, num_dual_iterations):

        assert(var_lp_f.shape[1] == self.num_var_lp_f)
        assert(con_lp_f.shape[1] == self.num_con_lp_f)
        assert(edge_lp_f_wo_ss.shape[1] == self.num_edge_lp_f_with_ss - 2)
        for d in range(len(self.feature_refinement)):
            var_learned_f, con_learned_f, edge_learned_f = self.feature_refinement[d](
                    var_learned_f, var_lp_f, con_learned_f, con_lp_f, edge_learned_f, solver_state, edge_lp_f_wo_ss, edge_index_var_con
                )

        prediction = self.predictor(
            torch.cat((var_learned_f, var_lp_f), 1), torch.cat((con_learned_f, con_lp_f), 1),
            torch.cat((edge_learned_f, solver_state['lo_costs'].unsqueeze(1), solver_state['hi_costs'].unsqueeze(1), edge_lp_f_wo_ss), 1), 
            edge_index_var_con).squeeze()

        dist_weights = prediction[:, 0]
        damping = torch.sigmoid(prediction[:, 1] - 3.0)
        # Project dist_weights to simplex:
        dist_weights = sol_utils.normalize_distribution_weights_softmax(dist_weights, edge_index_var_con)
        try:
            assert(torch.all(torch.isfinite(dist_weights)))
        except:
            breakpoint()

        var_indices = edge_index_var_con[0, :]
        new_mm = edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('current_mm_diff')].clone()
        solvers, solver_state, new_mm, new_lb = run_full_coord_iterations(solvers, solver_state['lo_costs'], solver_state['hi_costs'], solver_state['def_mm'], new_mm, damping, dist_weights, var_indices, num_dual_iterations)

        # Update per BDD solution:
        with torch.no_grad(): #TODO: use black-box backprop?
            edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('sol')] = sol_utils.compute_per_bdd_solution(solvers, solver_state)

        # Update lower bound for each BDD:
        con_lp_f[:, self.con_lp_f_names.index('lb')] = new_lb
        edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('current_mm_diff')] = new_mm
        return solver_state, var_lp_f, con_lp_f, edge_lp_f_wo_ss, dist_weights, damping
