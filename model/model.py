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
                num_edge_lp_f, in_edge_dim, out_edge_dim,
                use_layer_norm, use_skip_connections = False):
        super(FeatureExtractorLayer, self).__init__()

        self.con_updater = TransformerConv((num_var_lp_f + in_var_dim, num_con_lp_f + in_con_dim), 
                                            out_con_dim, 
                                            edge_dim = num_edge_lp_f + in_edge_dim, 
                                            flow="source_to_target", aggr = 'mean')

        self.var_updater = TransformerConv((out_con_dim + num_con_lp_f, num_var_lp_f + in_var_dim), 
                                            out_var_dim,
                                            edge_dim = num_edge_lp_f + in_edge_dim, 
                                            flow="target_to_source", aggr = 'mean')

        self.edge_updater = EdgeUpdater(num_edge_lp_f + in_edge_dim, out_edge_dim, out_var_dim + num_var_lp_f, out_con_dim + num_con_lp_f)
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

    def forward(self, var_learned_f, var_lp_f, con_learned_f, con_lp_f, edge_learned_f, solver_state, edge_rest_lp_f, edge_index_var_con):
        # 0. Combine learned and fixed features:
        var_comb_f = self.combine_features(var_learned_f, var_lp_f)
        con_comb_f = self.combine_features(con_learned_f, con_lp_f)
        edge_comb_f = torch.cat((edge_learned_f, solver_state['lo_costs'].unsqueeze(1), solver_state['hi_costs'].unsqueeze(1), solver_state['def_mm'].unsqueeze(1), edge_rest_lp_f), 1)
        # edge_comb_f_valid = edge_comb_f[valid_edge_mask, :]
        # edge_index_var_con_valid = edge_index_var_con[:, valid_edge_mask]
        
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
    def __init__(self, num_var_lp_f, out_var_dim, num_con_lp_f, out_con_dim, num_edge_lp_f, out_edge_dim, depth, use_layer_norm = False, skip_connections = False):
        super(FeatureExtractor, self).__init__()
        self.num_var_lp_f = num_var_lp_f
        self.num_con_lp_f = num_con_lp_f
        self.num_edge_lp_f = num_edge_lp_f
        self.skip_connections = skip_connections
        layers = [
            FeatureExtractorLayer(num_var_lp_f, 0, out_var_dim,
                                num_con_lp_f, 0, out_con_dim,
                                num_edge_lp_f, 0, out_edge_dim,
                                use_layer_norm, False)
        ]
        for l in range(depth - 1):
            layers.append(
                FeatureExtractorLayer(num_var_lp_f, out_var_dim, out_var_dim,
                                    num_con_lp_f, out_con_dim, out_con_dim,
                                    num_edge_lp_f, out_edge_dim, out_edge_dim,
                                    use_layer_norm, skip_connections)
            )
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, var_lp_f, con_lp_f, solver_state, edge_rest_lp_f, edge_index_var_con):
        assert(var_lp_f.shape[1] == self.num_var_lp_f)
        assert(con_lp_f.shape[1] == self.num_con_lp_f)
        assert(edge_rest_lp_f.shape[1] == self.num_edge_lp_f - 3) # 3 for solver state.
        #var_degree = var_lp_f[:, 1]
        #valid_edge_mask = get_edge_mask_without_terminal_nodes(edge_index_var_con, var_degree)
        var_learned_f = torch.zeros((var_lp_f.shape[0], 0), device = var_lp_f.device)
        con_learned_f = torch.zeros((con_lp_f.shape[0], 0), device = con_lp_f.device)
        edge_learned_f = torch.zeros((edge_rest_lp_f.shape[0], 0), device = edge_rest_lp_f.device)

        for l in range(len(self.layers)):
            var_learned_f, con_learned_f, edge_learned_f = self.layers[l](
                var_learned_f, var_lp_f, con_learned_f, con_lp_f, edge_learned_f, solver_state, edge_rest_lp_f,
                edge_index_var_con)
            
        
        return var_learned_f, con_learned_f, edge_learned_f

class PrimalPerturbationBlock(torch.nn.Module):
    def __init__(self, num_var_lp_f, num_con_lp_f, num_edge_lp_f, depth, var_dim, con_dim, edge_dim, use_layer_norm = False, predict_dist_weights = False, skip_connections = False):
        super(PrimalPerturbationBlock, self).__init__()
        self.feature_refinement = []
        for d in range(depth):
            self.feature_refinement.append(FeatureExtractorLayer(num_var_lp_f, var_dim, var_dim,
                                                                num_con_lp_f, con_dim, con_dim,   
                                                                num_edge_lp_f, edge_dim, edge_dim,
                                                                use_layer_norm, skip_connections))

        self.feature_refinement = torch.nn.ModuleList(self.feature_refinement)
        self.pert_predictor = nn.Sequential(nn.Linear(num_var_lp_f + var_dim + 1, num_var_lp_f + var_dim),
                                            nn.ReLU(True),
                                            nn.Linear(num_var_lp_f + var_dim, 1))
        if predict_dist_weights:
            self.dist_weights_predictor = nn.Sequential(nn.Linear(edge_dim + num_edge_lp_f - 3, edge_dim),
                                                nn.ReLU(True),
                                                nn.Linear(edge_dim, 1))
        else:
            self.dist_weights_predictor = None

        # self.mm_multiplier = nn.Parameter(torch.ones(1))
        # self.pert_multiplier = nn.Parameter(torch.ones(1) * 1e-3)

    def forward(self, solvers, var_lp_f, con_lp_f, 
                solver_state, edge_rest_lp_f, 
                var_learned_f, con_learned_f, edge_learned_f, 
                dist_weights, omega, edge_index_var_con,
                num_dual_iterations, grad_dual_itr_max_itr, dual_improvement_slope,
                batch_index_var):

        try:
            assert(torch.all(torch.isfinite(var_learned_f)))
            assert(torch.all(torch.isfinite(con_learned_f)))
            assert(torch.all(torch.isfinite(edge_learned_f)))
        except:
            breakpoint()

        #var_degree = var_lp_f[:, 1]
        #valid_edge_mask = get_edge_mask_without_terminal_nodes(edge_index_var_con, var_degree)
        for d in range(len(self.feature_refinement)):
            var_learned_f, con_learned_f, edge_learned_f = self.feature_refinement[d](
                    var_learned_f, var_lp_f, con_learned_f, con_lp_f, edge_learned_f, solver_state, edge_rest_lp_f, edge_index_var_con
                )
            try:
                assert(torch.all(torch.isfinite(var_learned_f)))
                assert(torch.all(torch.isfinite(con_learned_f)))
                assert(torch.all(torch.isfinite(edge_learned_f)))
            except:
                breakpoint()

        var_indices = edge_index_var_con[0]
        mm_diff_sum = scatter_sum(edge_rest_lp_f[:, 2], var_indices)
        primal_perturbation = self.pert_predictor(torch.cat((var_learned_f, var_lp_f, mm_diff_sum.unsqueeze(1)), 1)).squeeze() 

        net_primal_cost = var_lp_f[:, 0] + primal_perturbation
        # Make net_primal_cost have L2 norm = 1:
        normalized_net_primal_cost, _  = sol_utils.normalize_objective_batch(net_primal_cost, batch_index_var)
        
        # Zero-out current primal objective in var_lp_f[:, 2] and add net primal cost.
        normalized_net_primal_pert = normalized_net_primal_cost - var_lp_f[:, 0]

        try:
            assert(torch.all(torch.isfinite(normalized_net_primal_pert)))
        except:
            breakpoint()

        solver_state['lo_costs'], solver_state['hi_costs'] = sol_utils.perturb_primal_costs(solver_state['lo_costs'], solver_state['hi_costs'], normalized_net_primal_pert, edge_index_var_con)

        # Dual iterations
        if self.dist_weights_predictor is not None:
            dist_weights = sol_utils.normalize_distribution_weights_softmax(self.dist_weights_predictor(torch.cat((edge_learned_f, edge_rest_lp_f), 1)).squeeze(), edge_index_var_con)
        solver_state = sol_utils.dual_iterations(solvers, solver_state, dist_weights, num_dual_iterations, omega, dual_improvement_slope, grad_dual_itr_max_itr)
        solver_state = sol_utils.distribute_delta(solvers, solver_state)

        var_lp_f[:, 0] = normalized_net_primal_cost
        edge_rest_lp_f[:, 2] = sol_utils.compute_all_min_marginal_diff(solvers, solver_state)
        # Update per BDD solution:
        with torch.no_grad(): #TODO: use black-box backprop?
            new_lb = sol_utils.compute_per_bdd_lower_bound(solvers, solver_state)
            con_lp_f[:, 0] = new_lb # Need backprop through lb ?
            edge_rest_lp_f[:, 0] = sol_utils.compute_per_bdd_solution(solvers, solver_state)
       
        return solver_state, var_lp_f, con_lp_f, edge_rest_lp_f

class DualDistWeightsBlock(torch.nn.Module):
    def __init__(self, num_var_lp_f, num_con_lp_f, num_edge_lp_f, depth, var_dim, con_dim, edge_dim, use_layer_norm = False, predict_omega = False, skip_connections = False):
        super(DualDistWeightsBlock, self).__init__()
        self.feature_refinement = []
        for d in range(depth):
            self.feature_refinement.append(FeatureExtractorLayer(num_var_lp_f, var_dim, var_dim,
                                                                num_con_lp_f, con_dim, con_dim,   
                                                                num_edge_lp_f, edge_dim, edge_dim,
                                                                use_layer_norm, skip_connections))

        self.feature_refinement = torch.nn.ModuleList(self.feature_refinement)
        self.predict_omega = predict_omega
        num_outputs = 1
        if predict_omega:
            num_outputs += 1
        self.dist_weights_predictor = EdgeUpdater(num_edge_lp_f + edge_dim, num_outputs, 
                                            num_var_lp_f + var_dim, 
                                            num_con_lp_f + con_dim)
        
    def forward(self, solvers, var_lp_f, con_lp_f, 
                solver_state, edge_rest_lp_f, 
                var_learned_f, con_learned_f, edge_learned_f, 
                omega, edge_index_var_con, #TODOAA: Learn omega?
                num_dual_iterations, grad_dual_itr_max_itr, dual_improvement_slope, valid_edge_mask, batch_index_edge):
        for d in range(len(self.feature_refinement)):
            var_learned_f, con_learned_f, edge_learned_f = self.feature_refinement[d](
                    var_learned_f, var_lp_f, con_learned_f, con_lp_f, edge_learned_f, solver_state, edge_rest_lp_f, edge_index_var_con
                )

        predictions = self.dist_weights_predictor(
            torch.cat((var_learned_f, var_lp_f), 1), torch.cat((con_learned_f, con_lp_f), 1),
            torch.cat((edge_learned_f, solver_state['lo_costs'].unsqueeze(1), solver_state['hi_costs'].unsqueeze(1), solver_state['def_mm'].unsqueeze(1), edge_rest_lp_f), 1), 
            edge_index_var_con) #+ 1e-6
        dist_weights = predictions[:, 0]
        if self.predict_omega:
            omega = torch.sigmoid(scatter_mean(predictions[:, 1], batch_index_edge))

        #print(f'Predicted dist_weights, min: {dist_weights[valid_edge_mask].min()}, max: {dist_weights[valid_edge_mask].max()}')

        # Project dist_weights to simplex:
        #dist_weights = sol_utils.normalize_distribution_weights(dist_weights, edge_index_var_con)
        dist_weights = sol_utils.normalize_distribution_weights_softmax(dist_weights, edge_index_var_con)
        try:
            assert(torch.all(torch.isfinite(dist_weights)))
        except:
            breakpoint()

        # Dual iterations
        solver_state = sol_utils.dual_iterations(solvers, solver_state, dist_weights, num_dual_iterations, omega, dual_improvement_slope, grad_dual_itr_max_itr)

        edge_rest_lp_f[:, 2] = dist_weights
        con_lp_f[:, 0] = sol_utils.compute_per_bdd_lower_bound(solvers, solver_state)
        # Update per BDD solution:
        with torch.no_grad(): #TODO: use black-box backprop?
            edge_rest_lp_f[:, 0] = sol_utils.compute_per_bdd_solution(solvers, solver_state)
       
        return solver_state, var_lp_f, con_lp_f, edge_rest_lp_f, dist_weights

class DualFullCoordinateAscent(torch.nn.Module):
    def __init__(self, num_var_lp_f, num_con_lp_f, num_edge_lp_f, depth, var_dim, con_dim, edge_dim, use_layer_norm = False, skip_connections = False):
        super(DualFullCoordinateAscent, self).__init__()
        self.feature_refinement = []
        for d in range(depth):
            self.feature_refinement.append(FeatureExtractorLayer(num_var_lp_f, var_dim, var_dim,
                                                                num_con_lp_f, con_dim, con_dim,   
                                                                num_edge_lp_f, edge_dim, edge_dim,
                                                                use_layer_norm, skip_connections))

        self.feature_refinement = torch.nn.ModuleList(self.feature_refinement)
        self.predictor = EdgeUpdater(num_edge_lp_f + edge_dim, 2, 
                                            num_var_lp_f + var_dim, 
                                            num_con_lp_f + con_dim)
    
    def forward(self, solvers, var_lp_f, con_lp_f, 
                solver_state, edge_rest_lp_f, 
                var_learned_f, con_learned_f, edge_learned_f, 
                edge_index_var_con, num_dual_iterations):

        for d in range(len(self.feature_refinement)):
            var_learned_f, con_learned_f, edge_learned_f = self.feature_refinement[d](
                    var_learned_f, var_lp_f, con_learned_f, con_lp_f, edge_learned_f, solver_state, edge_rest_lp_f, edge_index_var_con
                )

        prediction = self.predictor(
            torch.cat((var_learned_f, var_lp_f), 1), torch.cat((con_learned_f, con_lp_f), 1),
            torch.cat((edge_learned_f, solver_state['lo_costs'].unsqueeze(1), solver_state['hi_costs'].unsqueeze(1), solver_state['def_mm'].unsqueeze(1), edge_rest_lp_f), 1), 
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
        new_mm = edge_rest_lp_f[:, 3].clone()
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
            edge_rest_lp_f[:, 0] = sol_utils.compute_per_bdd_solution(solvers, solver_state)

        # Update lower bound for each BDD:
        con_lp_f[:, 0] = sol_utils.compute_per_bdd_lower_bound(solvers, solver_state)

        edge_rest_lp_f[:, 3] = new_mm
        return solver_state, var_lp_f, con_lp_f, edge_rest_lp_f
