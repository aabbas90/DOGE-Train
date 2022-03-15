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
                use_def_mm = True):
        super(FeatureExtractorLayer, self).__init__()

        self.use_def_mm = use_def_mm
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
        if self.use_def_mm:
            edge_comb_f = torch.cat((edge_learned_f, solver_state['lo_costs'].unsqueeze(1), solver_state['hi_costs'].unsqueeze(1), solver_state['def_mm'].unsqueeze(1), edge_lp_f_wo_ss), 1)
        else:
            edge_comb_f = torch.cat((edge_learned_f, solver_state['lo_costs'].unsqueeze(1), solver_state['hi_costs'].unsqueeze(1), edge_lp_f_wo_ss), 1)
        
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
                use_layer_norm = False, predict_dist_weights = False, skip_connections = False):
        super(PrimalPerturbationBlock, self).__init__()
        assert(not predict_dist_weights)
        self.min_perturbation = min_perturbation
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
                                                                use_layer_norm, skip_connections, False))

        self.feature_refinement = torch.nn.ModuleList(self.feature_refinement)
        self.pert_predictor = nn.Sequential(nn.Linear(self.num_edge_lp_f_with_ss + edge_dim, self.num_edge_lp_f_with_ss + edge_dim),
                                            nn.ReLU(True),
                                            nn.Linear(self.num_edge_lp_f_with_ss + edge_dim, 1))
        if predict_dist_weights:
            self.dist_weights_predictor = nn.Sequential(nn.Linear(edge_dim + self.num_edge_lp_f_with_ss, edge_dim),
                                                nn.ReLU(True),
                                                nn.Linear(edge_dim, 1))
        else:
            self.dist_weights_predictor = None

    def forward(self, solvers, var_lp_f, con_lp_f, 
                solver_state, edge_lp_f_wo_ss, 
                var_learned_f, con_learned_f, edge_learned_f, 
                dist_weights, omega, edge_index_var_con,
                num_dual_iterations, grad_dual_itr_max_itr, dual_improvement_slope,
                batch_index_var, batch_index_con, batch_index_edge, norms):

        # First normalize input costs:
        prev_lb = con_lp_f[:, self.con_lp_f_names.index('new_lb')].clone()
        prev_mm_diff = edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('new_mm_diff')].clone()
        solver_state, normalized_lb, norm_mm_diff, norms = sol_utils.normalize_costs(solver_state, prev_lb, prev_mm_diff, norms, batch_index_edge, batch_index_con)

        con_lp_f[:, self.con_lp_f_names.index('new_lb')] = normalized_lb
        edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('new_mm_diff')] = norm_mm_diff
        try:
            assert(torch.all(torch.isfinite(var_learned_f)))
            assert(torch.all(torch.isfinite(con_learned_f)))
            assert(torch.all(torch.isfinite(edge_learned_f)))
        except:
            breakpoint()

        for d in range(len(self.feature_refinement)):
            var_learned_f, con_learned_f, edge_learned_f = self.feature_refinement[d](
                    var_learned_f, var_lp_f, con_learned_f, con_lp_f, edge_learned_f, solver_state, edge_lp_f_wo_ss, edge_index_var_con
                )
            try:
                assert(torch.all(torch.isfinite(var_learned_f)))
                assert(torch.all(torch.isfinite(con_learned_f)))
                assert(torch.all(torch.isfinite(edge_learned_f)))
            except:
                breakpoint()

        primal_perturbation = 0.1 * self.pert_predictor(torch.cat((edge_learned_f, edge_lp_f_wo_ss), 1)).squeeze()

        # Make net primal costs have L2 norm = 1:
        p_lo_cost = torch.relu(primal_perturbation + self.min_perturbation * 0.5)
        p_hi_cost = torch.relu(-primal_perturbation + self.min_perturbation * 0.5)
        # norm_costs_batch = torch.sqrt(scatter_sum(torch.square(new_hi_cost), batch_index_var)) + torch.sqrt(scatter_sum(torch.square(new_lo_cost), batch_index_var))
        
        # normalized_lo_cost = new_lo_cost / norm_costs_batch[batch_index_var]
        # normalized_hi_cost = new_hi_cost / norm_costs_batch[batch_index_var]
    
        # Zero-out current primal objective in var_lp_f[:, 0] and add net primal cost.
        # normalized_net_lo_cost_pert = normalized_lo_cost - var_lp_f[:, 0]
        # normalized_net_hi_cost_pert = normalized_hi_cost - var_lp_f[:, 1]
        # var_lp_f[:, 0] = normalized_net_lo_cost_pert
        # var_lp_f[:, 1] = normalized_net_hi_cost_pert

        try:
            assert(torch.all(torch.isfinite(primal_perturbation)))
        except:
            breakpoint()

        #solver_state['lo_costs'], solver_state['hi_costs'] = sol_utils.perturb_primal_costs(solver_state['lo_costs'], solver_state['hi_costs'], normalized_net_lo_cost_pert, normalized_net_hi_cost_pert, dist_weights, edge_index_var_con)
        solver_state['lo_costs'] = solver_state['lo_costs'] + p_lo_cost
        solver_state['hi_costs'] = solver_state['hi_costs'] + p_hi_cost

        # Dual iterations
        if self.dist_weights_predictor is not None:
            dist_weights = sol_utils.normalize_distribution_weights_softmax(self.dist_weights_predictor(torch.cat((edge_learned_f, edge_lp_f_wo_ss), 1)).squeeze(), edge_index_var_con)
        solver_state = sol_utils.dual_iterations(solvers, solver_state, dist_weights, num_dual_iterations, omega, dual_improvement_slope, grad_dual_itr_max_itr)
        solver_state = sol_utils.distribute_delta(solvers, solver_state)

        edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('new_mm_diff')] = sol_utils.compute_all_min_marginal_diff(solvers, solver_state)
        con_lp_f[:, self.con_lp_f_names.index('new_lb')] = sol_utils.compute_per_bdd_lower_bound(solvers, solver_state, True) # Update perturbed lower bound.

        # Update per BDD solution:
        with torch.no_grad(): #TODO: use black-box backprop?
            edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('sol')] = sol_utils.compute_per_bdd_solution(solvers, solver_state)
       
        return solver_state, var_lp_f, con_lp_f, edge_lp_f_wo_ss, norms

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
                                                                use_layer_norm, skip_connections))

        self.feature_refinement = torch.nn.ModuleList(self.feature_refinement)
        self.predictor = EdgeUpdater(self.num_edge_lp_f_with_ss + edge_dim, 2, self.num_var_lp_f + var_dim, self.num_con_lp_f + con_dim)
    
    def forward(self, solvers, var_lp_f, con_lp_f, 
                solver_state, edge_lp_f_wo_ss, 
                var_learned_f, con_learned_f, edge_learned_f, 
                edge_index_var_con, num_dual_iterations):

        for d in range(len(self.feature_refinement)):
            var_learned_f, con_learned_f, edge_learned_f = self.feature_refinement[d](
                    var_learned_f, var_lp_f, con_learned_f, con_lp_f, edge_learned_f, solver_state, edge_lp_f_wo_ss, edge_index_var_con
                )

        #TODO Ensure def_mm = 0.
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
        new_mm = edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('prev_mm_diff')].clone()
        for dual_itr in range(num_dual_iterations):
            incoming_message = damping * new_mm
            net_delta_edge = scatter_sum(incoming_message, var_indices)[var_indices] * dist_weights - incoming_message

            # Update costs:
            # solver_state['lo_costs'] = solver_state['lo_costs'] + torch.relu(-net_delta_edge) # lo cost do we need to update lo costs in full coordinate ascent?
            solver_state['hi_costs'] = solver_state['hi_costs'] + net_delta_edge  # corresponds directly to lagrange multipliers.
    
            # Compute new min-marginal differences:
            new_mm = sol_utils.compute_all_min_marginal_diff(solvers, solver_state)
            #TODO Put a NN here?

        # Update per BDD solution:
        with torch.no_grad(): #TODO: use black-box backprop?
            edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('sol')] = sol_utils.compute_per_bdd_solution(solvers, solver_state)

        # Update lower bound for each BDD:
        con_lp_f[:, self.con_lp_f_names.index('lb')] = sol_utils.compute_per_bdd_lower_bound(solvers, solver_state)
        edge_lp_f_wo_ss[:, self.edge_lp_f_names.index('prev_mm_diff')] = new_mm
        return solver_state, var_lp_f, con_lp_f, edge_lp_f_wo_ss
