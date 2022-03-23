import torch
import torch.nn as nn
from torch_scatter import scatter_sum, scatter_mean
from torch_geometric.nn import TransformerConv, GATv2Conv, LayerNorm
import model.solver_utils as sol_utils

# https://github.com/rusty1s/pytorch_geometric/issues/1210 might be useful
# https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html?highlight=meta%20layer#torch_geometric.nn.meta.MetaLayer
class EdgeUpdater(torch.nn.Module):
    # Receives learned var, con, edge features and fixed edge features to predict new learned edge features.
    def __init__(self, num_input_edge_channels, num_output_edge_channels, num_var_channels, num_con_channels):
        super(EdgeUpdater, self).__init__()
        # num_input_channels = num_var_channels + num_con_channels + num_input_edge_channels
        self.var_compress_mlp = nn.Sequential(nn.Linear(num_var_channels, num_output_edge_channels), nn.ReLU(True),
                                        nn.Linear(num_output_edge_channels, num_output_edge_channels), nn.ReLU(True))

        self.con_compress_mlp = nn.Sequential(nn.Linear(num_con_channels, num_output_edge_channels), nn.ReLU(True),
                                        nn.Linear(num_output_edge_channels, num_output_edge_channels), nn.ReLU(True))

        self.edge_mlp = nn.Sequential(nn.Linear(num_input_edge_channels + num_output_edge_channels + num_output_edge_channels, num_output_edge_channels),
                                        nn.ReLU(True),
                                        nn.Linear(num_output_edge_channels, num_output_edge_channels))

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

class FeatureExtractorLayer(torch.nn.Module):
    def __init__(self, 
                num_var_lp_f, in_var_dim, out_var_dim, 
                num_con_lp_f, in_con_dim, out_con_dim,
                num_edge_lp_f_with_ss, in_edge_dim, out_edge_dim,
                use_layer_norm, use_skip_connections = False, 
                use_def_mm = True, use_solver_costs = True):
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

        self.edge_updater = EdgeUpdater(num_edge_lp_f_with_ss + in_edge_dim, out_edge_dim, out_var_dim + num_var_lp_f, out_con_dim + num_con_lp_f)
        self.use_skip_connections = use_skip_connections

        if use_layer_norm:
            self.var_norm = LayerNorm(out_var_dim)
            self.con_norm = LayerNorm(out_con_dim)
            self.edge_norm = LayerNorm(out_edge_dim)
        else:
            self.var_norm = torch.nn.Identity()
            self.con_norm = torch.nn.Identity()
            self.edge_norm = torch.nn.Identity()

    def combine_features(self, f1, f2):
        return torch.cat((f1, f2), 1)

    def forward(self, var_learned_f, var_lp_f, con_learned_f, con_lp_f, edge_learned_f, solver_state, edge_lp_f_wo_ss, edge_index_var_con):
        # 0. Combine learned and fixed features:
        var_comb_f = self.combine_features(var_learned_f, var_lp_f)
        con_comb_f = self.combine_features(con_learned_f, con_lp_f)
        if self.use_solver_costs:
            if self.use_def_mm:
                edge_comb_f = torch.cat((edge_learned_f, solver_state['lo_costs'].unsqueeze(1), solver_state['hi_costs'].unsqueeze(1), solver_state['def_mm'].unsqueeze(1), edge_lp_f_wo_ss), 1)
            else:
                edge_comb_f = torch.cat((edge_learned_f, solver_state['lo_costs'].unsqueeze(1), solver_state['hi_costs'].unsqueeze(1), edge_lp_f_wo_ss), 1)
        else:
            assert(not self.use_def_mm)        
            edge_comb_f = torch.cat((edge_learned_f, edge_lp_f_wo_ss), 1)

        # 1. Update constraint features (var, constraint, edge -> constraint_new):
        if self.use_skip_connections:
            con_learned_f = torch.relu(self.con_norm(con_learned_f + self.con_updater((var_comb_f, con_comb_f), edge_index_var_con, edge_comb_f)))
        else:
            con_learned_f = torch.relu(self.con_norm(self.con_updater((var_comb_f, con_comb_f), edge_index_var_con, edge_comb_f)))

        con_comb_f = self.combine_features(con_learned_f, con_lp_f)

        # 2. Update variable features (var, constraint_new, edge -> var):
        if self.use_skip_connections:
            var_learned_f = torch.relu(self.var_norm(var_learned_f + self.var_updater((con_comb_f, var_comb_f), edge_index_var_con, edge_comb_f)))
        else:
            var_learned_f = torch.relu(self.var_norm(self.var_updater((con_comb_f, var_comb_f), edge_index_var_con, edge_comb_f)))

        var_comb_f = self.combine_features(var_learned_f, var_lp_f)

        # 3. Update edges:
        if self.use_skip_connections:
            edge_learned_f = torch.relu(self.edge_norm(edge_learned_f + self.edge_updater(var_comb_f, con_comb_f, edge_comb_f, edge_index_var_con)))
        else:
            edge_learned_f = torch.relu(self.edge_norm(self.edge_updater(var_comb_f, con_comb_f, edge_comb_f, edge_index_var_con)))

        return var_learned_f, con_learned_f, edge_learned_f

def get_edge_mask_without_terminal_nodes(edge_index_var_con, var_degree):
    var_indices = edge_index_var_con[0]
    non_terminal_vars = var_degree != 0
    non_terminal_edges_mask = non_terminal_vars[var_indices]
    return non_terminal_edges_mask

class FeatureExtractor(torch.nn.Module):
    def __init__(self, num_var_lp_f, out_var_dim, num_con_lp_f, out_con_dim, num_edge_lp_f, out_edge_dim, depth, 
                use_layer_norm = False, skip_connections = False, use_def_mm = True):
        super(FeatureExtractor, self).__init__()
        self.num_var_lp_f = num_var_lp_f
        self.num_con_lp_f = num_con_lp_f
        num_edge_lp_f_with_ss = num_edge_lp_f + 3 - int(not use_def_mm)
        self.skip_connections = skip_connections
        self.use_def_mm = use_def_mm
        layers = [
            FeatureExtractorLayer(num_var_lp_f, 0, out_var_dim,
                                num_con_lp_f, 0, out_con_dim,
                                num_edge_lp_f_with_ss, 0, out_edge_dim,
                                use_layer_norm, False, use_def_mm)
        ]
        for l in range(depth - 1):
            layers.append(
                FeatureExtractorLayer(num_var_lp_f, out_var_dim, out_var_dim,
                                    num_con_lp_f, out_con_dim, out_con_dim,
                                    num_edge_lp_f_with_ss, out_edge_dim, out_edge_dim,
                                    use_layer_norm, skip_connections, use_def_mm)
            )
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, var_lp_f, con_lp_f, solver_state, edge_lp_f_wo_ss, edge_index_var_con):
        assert(var_lp_f.shape[1] == self.num_var_lp_f)
        assert(con_lp_f.shape[1] == self.num_con_lp_f)
        #var_degree = var_lp_f[:, 1]
        #valid_edge_mask = get_edge_mask_without_terminal_nodes(edge_index_var_con, var_degree)
        var_learned_f = torch.zeros((var_lp_f.shape[0], 0), device = var_lp_f.device)
        con_learned_f = torch.zeros((con_lp_f.shape[0], 0), device = con_lp_f.device)
        edge_learned_f = torch.zeros((edge_lp_f_wo_ss.shape[0], 0), device = edge_lp_f_wo_ss.device)

        for l in range(len(self.layers)):
            var_learned_f, con_learned_f, edge_learned_f = self.layers[l](
                var_learned_f, var_lp_f, con_learned_f, con_lp_f, edge_learned_f, solver_state, edge_lp_f_wo_ss,
                edge_index_var_con)
            
        
        return var_learned_f, con_learned_f, edge_learned_f

class PrimalPerturbationBlock(torch.nn.Module):
    def __init__(self, var_lp_f_names, con_lp_f_names, edge_lp_f_names, depth, var_dim, con_dim, edge_dim, min_perturbation, 
                use_layer_norm = False, skip_connections = False, use_lstm_var = False):
        super(PrimalPerturbationBlock, self).__init__()
        self.min_perturbation = min_perturbation
        self.var_lp_f_names = var_lp_f_names
        self.con_lp_f_names = con_lp_f_names
        self.edge_lp_f_names = edge_lp_f_names
        self.num_var_lp_f = len(var_lp_f_names)
        self.num_con_lp_f = len(con_lp_f_names)
        self.num_edge_lp_f_with_ss = len(edge_lp_f_names)
        self.feature_refinement = []
        for d in range(depth):
            self.feature_refinement.append(FeatureExtractorLayer(self.num_var_lp_f, var_dim, var_dim,
                                                                self.num_con_lp_f, con_dim, con_dim,   
                                                                self.num_edge_lp_f_with_ss, edge_dim, edge_dim,
                                                                use_layer_norm, skip_connections, False, False))

        self.feature_refinement = torch.nn.ModuleList(self.feature_refinement)
        self.var_lstm = None
        if use_lstm_var:
            self.var_lstm = nn.LSTMCell(var_dim + 1, var_dim)
        self.var_pert_predictor = nn.Sequential(nn.Linear(var_dim, var_dim), nn.ReLU(True), nn.Linear(var_dim, var_dim), nn.ReLU(True), nn.Linear(var_dim, 1))
        self.tanh_softness = torch.nn.Parameter(torch.ones(1) * 10.0)
        self.learned_pert_influence = torch.nn.Parameter(torch.ones(1) * 0.0001)
        self.mm_sign_influence = torch.nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, solvers, var_lp_f, con_lp_f, 
                orig_solver_state, edge_lp_f_wo_ss, 
                var_learned_f, con_learned_f, edge_learned_f, 
                dist_weights, omega, edge_index_var_con,
                num_dual_iterations, grad_dual_itr_max_itr, dual_improvement_slope,
                batch_index_var, batch_index_con, batch_index_edge, num_bdds_per_inst,
                var_hidden_states_lstm = None):

        # perturb orig_solver_state by knowing prev_solver_state.

        # First normalize costs in prev_solver_state as they will be sent to GNN:
        prev_var_costs = var_lp_f[:, self.var_lp_f_names.index('prev_obj')].clone()
        prev_lb_per_bdd = con_lp_f[:, self.con_lp_f_names.index('prev_lb')].clone()
        prev_mm_diff = edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('prev_mm_diff')].clone()
        prev_var_costs, prev_lb_per_bdd, prev_mm_diff, _, _ = sol_utils.normalize_costs_var(
                prev_var_costs, prev_lb_per_bdd, prev_mm_diff, num_bdds_per_inst, batch_index_var, batch_index_con, batch_index_edge)

        var_lp_f[:, self.var_lp_f_names.index('prev_obj')] = prev_var_costs
        con_lp_f[:, self.con_lp_f_names.index('prev_lb')] = prev_lb_per_bdd
        edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('prev_mm_diff')] = prev_mm_diff
        try:
            assert(torch.all(torch.isfinite(var_learned_f)))
            assert(torch.all(torch.isfinite(con_learned_f)))
            assert(torch.all(torch.isfinite(edge_learned_f)))
        except:
            breakpoint()

        for d in range(len(self.feature_refinement)):
            var_learned_f, con_learned_f, edge_learned_f = self.feature_refinement[d](
                    var_learned_f, var_lp_f, con_learned_f, con_lp_f, edge_learned_f, None, edge_lp_f_wo_ss, edge_index_var_con
                )
            try:
                assert(torch.all(torch.isfinite(var_learned_f)))
                assert(torch.all(torch.isfinite(con_learned_f)))
                assert(torch.all(torch.isfinite(edge_learned_f)))
            except:
                breakpoint()

        var_indices = edge_index_var_con[0]
        prev_mm_diff = edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('prev_mm_diff')].clone()
        mm_diff_soft_sign = scatter_mean(torch.tanh(self.tanh_softness * prev_mm_diff), var_indices)
        var_learned_f_with_mm = torch.cat((var_learned_f, mm_diff_soft_sign.unsqueeze(1)), 1)
        if self.var_lstm is not None:
            assert(var_hidden_states_lstm is not None)
            var_hidden_states_lstm['h'], var_hidden_states_lstm['c'] = self.var_lstm(var_learned_f_with_mm, (var_hidden_states_lstm['h'], var_hidden_states_lstm['c']))
            primal_perturbation_var = self.var_pert_predictor(var_hidden_states_lstm['h']).squeeze()
        else:
            primal_perturbation_var = self.var_pert_predictor(var_learned_f).squeeze()

        primal_perturbation_var = torch.abs(self.learned_pert_influence) * primal_perturbation_var + torch.abs(self.mm_sign_influence) * mm_diff_soft_sign
        primal_perturbation_var_lo = -primal_perturbation_var + self.min_perturbation * 0.5
        primal_perturbation_var_hi = primal_perturbation_var + self.min_perturbation * 0.5

        p_lo_cost_var = torch.relu(primal_perturbation_var_lo)
        p_hi_cost_var = torch.relu(primal_perturbation_var_hi)
        
        p_lo_cost = p_lo_cost_var[var_indices] * dist_weights
        p_hi_cost = p_hi_cost_var[var_indices] * dist_weights

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

        edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('prev_mm_diff')] = sol_utils.compute_all_min_marginal_diff(solvers, pert_solver_state)
        con_lp_f[:, self.con_lp_f_names.index('prev_lb')] = sol_utils.compute_per_bdd_lower_bound(solvers, pert_solver_state) # Update perturbed lower bound.
        prev_var_costs = scatter_sum(pert_solver_state['hi_costs'] - pert_solver_state['lo_costs'], var_indices) # Convert to variable costs.
        var_lp_f[:, self.var_lp_f_names.index('prev_obj')] = prev_var_costs

        # Update per BDD solution:
        with torch.no_grad(): #TODO: use black-box backprop?
            edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('prev_sol')] = sol_utils.compute_per_bdd_solution(solvers, pert_solver_state)
       
        return var_lp_f, con_lp_f, edge_lp_f_wo_ss, var_hidden_states_lstm

class DualDistWeightsBlock(torch.nn.Module):
    def __init__(self, var_lp_f_names, con_lp_f_names, edge_lp_f_names, depth, var_dim, con_dim, edge_dim, use_layer_norm = False, predict_omega = False, skip_connections = False):
        super(DualDistWeightsBlock, self).__init__()
        self.var_lp_f_names = var_lp_f_names
        self.con_lp_f_names = con_lp_f_names
        self.edge_lp_f_names = edge_lp_f_names
        self.num_var_lp_f = len(var_lp_f_names)
        self.num_con_lp_f = len(con_lp_f_names)
        self.num_edge_lp_f_with_ss = len(edge_lp_f_names) + 3
        self.feature_refinement = []
        for d in range(depth):
            self.feature_refinement.append(FeatureExtractorLayer(self.num_var_lp_f, var_dim, var_dim,
                                                                self.num_con_lp_f, con_dim, con_dim,   
                                                                self.num_edge_lp_f_with_ss, edge_dim, edge_dim,
                                                                use_layer_norm, skip_connections))

        self.feature_refinement = torch.nn.ModuleList(self.feature_refinement)
        self.predict_omega = predict_omega
        num_outputs = 1
        if predict_omega:
            num_outputs += 1
        self.dist_weights_predictor = EdgeUpdater(self.num_edge_lp_f_with_ss + edge_dim, num_outputs, 
                                            self.num_var_lp_f + var_dim, 
                                            self.num_con_lp_f + con_dim)

    def forward(self, solvers, var_lp_f, con_lp_f, 
                solver_state, edge_lp_f_wo_ss, 
                var_learned_f, con_learned_f, edge_learned_f, 
                omega, edge_index_var_con,
                num_dual_iterations, grad_dual_itr_max_itr, dual_improvement_slope, valid_edge_mask, batch_index_edge):

        for d in range(len(self.feature_refinement)):
            var_learned_f, con_learned_f, edge_learned_f = self.feature_refinement[d](
                    var_learned_f, var_lp_f, con_learned_f, con_lp_f, edge_learned_f, solver_state, edge_lp_f_wo_ss, edge_index_var_con
                )

        predictions = self.dist_weights_predictor(
            torch.cat((var_learned_f, var_lp_f), 1), torch.cat((con_learned_f, con_lp_f), 1),
            torch.cat((edge_learned_f, solver_state['lo_costs'].unsqueeze(1), solver_state['hi_costs'].unsqueeze(1), solver_state['def_mm'].unsqueeze(1), edge_lp_f_wo_ss), 1), 
            edge_index_var_con) #+ 1e-6
        dist_weights = predictions[:, 0]
        dist_weights = sol_utils.normalize_distribution_weights_softmax(dist_weights, edge_index_var_con)
        try:
            assert(torch.all(torch.isfinite(dist_weights)))
        except:
            breakpoint()

        if self.predict_omega:
            omega_vec = torch.sigmoid(predictions[:, 1])
            # Dual iterations
            solver_state = sol_utils.dual_iterations(solvers, solver_state, dist_weights, num_dual_iterations, omega_vec, dual_improvement_slope, grad_dual_itr_max_itr)
        else:
            omega_vec = None
            # Dual iterations
            solver_state = sol_utils.dual_iterations(solvers, solver_state, dist_weights, num_dual_iterations, omega, dual_improvement_slope, grad_dual_itr_max_itr)

        edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('prev_dist_weights')] = dist_weights
        con_lp_f[:, self.con_lp_f_names.index('lb')] = sol_utils.compute_per_bdd_lower_bound(solvers, solver_state)
        # Update per BDD solution:
        with torch.no_grad(): #TODO: use black-box backprop?
            edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('sol')] = sol_utils.compute_per_bdd_solution(solvers, solver_state)
       
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
    def __init__(self, var_lp_f_names, con_lp_f_names, edge_lp_f_names, depth, var_dim, con_dim, edge_dim, use_layer_norm = False, skip_connections = False):
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
                                                                use_layer_norm, skip_connections, use_def_mm=False))

        self.feature_refinement = torch.nn.ModuleList(self.feature_refinement)
        self.predictor = EdgeUpdater(self.num_edge_lp_f_with_ss + edge_dim, 2, self.num_var_lp_f + var_dim, self.num_con_lp_f + con_dim)
    
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
