import torch
import torch.nn as nn
from torch_scatter import scatter_sum
from torch_geometric.nn import TransformerConv
import model.solver_utils as sol_utils

# https://github.com/rusty1s/pytorch_geometric/issues/1210 might be useful
# https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html?highlight=meta%20layer#torch_geometric.nn.meta.MetaLayer
class EdgeUpdater(torch.nn.Module):
    # Receives learned var, con, edge features and fixed edge features to predict new learned edge features.
    def __init__(self, num_input_edge_channels, num_output_edge_channels, num_var_channels, num_con_channels, post_activation = nn.ReLU()):
        super(EdgeUpdater, self).__init__()
        # num_input_channels = num_var_channels + num_con_channels + num_input_edge_channels
        self.post_activation = post_activation

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
        return self.post_activation(self.edge_mlp(out))

class FeatureExtractorLayer(torch.nn.Module):
    def __init__(self, 
                num_var_lp_f, in_var_dim, out_var_dim, 
                num_con_lp_f, in_con_dim, out_con_dim,
                num_edge_lp_f, in_edge_dim, out_edge_dim):
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

    def combine_features(self, f1, f2):
        return torch.cat((f1, f2), 1)

    def forward(self, var_learned_f, var_lp_f, con_learned_f, con_lp_f, edge_learned_f, solver_state, edge_rest_lp_f, edge_index_var_con):
        # 0. Combine learned and fixed features:
        var_comb_f = self.combine_features(var_learned_f, var_lp_f)
        con_comb_f = self.combine_features(con_learned_f, con_lp_f)
        edge_comb_f = torch.cat((edge_learned_f, solver_state['lo_costs'].unsqueeze(1), solver_state['hi_costs'].unsqueeze(1), solver_state['def_mm'].unsqueeze(1), edge_rest_lp_f), 1)

        # 1. Update constraint features (var, constraint, edge -> constraint_new):
        con_learned_f = torch.relu(self.con_updater((var_comb_f, con_comb_f), edge_index_var_con, edge_comb_f))
        con_comb_f = self.combine_features(con_learned_f, con_lp_f)

        # 2. Update variable features (var, constraint_new, edge -> var):
        var_learned_f = torch.relu(self.var_updater((con_comb_f, var_comb_f), edge_index_var_con, edge_comb_f))
        var_comb_f = self.combine_features(var_learned_f, var_lp_f)

        # 3. Update edges:
        edge_learned_f = self.edge_updater(var_comb_f, con_comb_f, edge_comb_f, edge_index_var_con)

        return var_learned_f, con_learned_f, edge_learned_f

class FeatureExtractor(torch.nn.Module):
    def __init__(self, num_var_lp_f, out_var_dim, num_con_lp_f, out_con_dim, num_edge_lp_f, out_edge_dim, depth):
        super(FeatureExtractor, self).__init__()
        self.num_var_lp_f = num_var_lp_f
        self.num_con_lp_f = num_con_lp_f
        self.num_edge_lp_f = num_edge_lp_f
        layers = [
            FeatureExtractorLayer(num_var_lp_f, 0, out_var_dim,
                                num_con_lp_f, 0, out_con_dim,
                                num_edge_lp_f, 0, out_edge_dim)
        ]
        for l in range(depth - 1):
            layers.append(
                FeatureExtractorLayer(num_var_lp_f, out_var_dim, out_var_dim,
                                    num_con_lp_f, out_con_dim, out_con_dim,
                                    num_edge_lp_f, out_edge_dim, out_edge_dim)
            )
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, var_lp_f, con_lp_f, solver_state, edge_rest_lp_f, edge_index_var_con):
        assert(var_lp_f.shape[1] == self.num_var_lp_f)
        assert(con_lp_f.shape[1] == self.num_con_lp_f)
        assert(edge_rest_lp_f.shape[1] == self.num_edge_lp_f - 3) # 3 for solver state.
        var_learned_f = torch.zeros((var_lp_f.shape[0], 0), device = var_lp_f.device)
        con_learned_f = torch.zeros((con_lp_f.shape[0], 0), device = con_lp_f.device)
        edge_learned_f = torch.zeros((edge_rest_lp_f.shape[0], 0), device = edge_rest_lp_f.device)

        for l in range(len(self.layers)):
            var_learned_f, con_learned_f, edge_learned_f = self.layers[l](
                var_learned_f, var_lp_f, con_learned_f, con_lp_f, edge_learned_f, solver_state, edge_rest_lp_f,
                edge_index_var_con) # TODO: Use residual connections?
        
        return var_learned_f, con_learned_f, edge_learned_f

class PrimalPerturbationBlock(torch.nn.Module):
    def __init__(self, num_var_lp_f, num_con_lp_f, num_edge_lp_f, depth, var_dim, con_dim, edge_dim):
        super(PrimalPerturbationBlock, self).__init__()
        self.feature_refinement = []
        for d in range(depth):
            self.feature_refinement.append(FeatureExtractorLayer(num_var_lp_f, var_dim, var_dim,
                                                                num_con_lp_f, con_dim, con_dim,   
                                                                num_edge_lp_f, edge_dim, edge_dim))

        self.feature_refinement = torch.nn.ModuleList(self.feature_refinement)
        self.edge_cost_predictor = EdgeUpdater(num_edge_lp_f + edge_dim, 1, 
                                            num_var_lp_f + var_dim, 
                                            num_con_lp_f + con_dim, 
                                            torch.nn.Identity())

    def forward(self, solvers, var_lp_f, con_lp_f, 
                solver_state, edge_rest_lp_f, 
                var_learned_f, con_learned_f, edge_learned_f, 
                dist_weights, omega, edge_index_var_con,
                num_dual_iterations, grad_dual_itr_max_itr, dual_improvement_slope):
        for d in range(len(self.feature_refinement)):
            var_learned_f, con_learned_f, edge_learned_f = self.feature_refinement[d](
                    var_learned_f, var_lp_f, con_learned_f, con_lp_f, edge_learned_f, solver_state, edge_rest_lp_f, edge_index_var_con
                )

        dual_cost_perturbation = 1e-1 * self.edge_cost_predictor(
            torch.cat((var_learned_f, var_lp_f), 1), torch.cat((con_learned_f, con_lp_f), 1),
            torch.cat((edge_learned_f, solver_state['lo_costs'].unsqueeze(1), solver_state['hi_costs'].unsqueeze(1), solver_state['def_mm'].unsqueeze(1), edge_rest_lp_f), 1), 
            edge_index_var_con).squeeze() 
            
        #TODO: Force dual_cost_perturbation sign to agree with mm diff:
        # dual_cost_perturbation = dual_cost_perturbation + edge_rest_lp_f[:, 2]

        try:
            assert(torch.all(torch.isfinite(dual_cost_perturbation)))
        except:
            breakpoint()
        solver_state['lo_costs'] = solver_state['lo_costs'] + torch.nn.ReLU(inplace=True)(-dual_cost_perturbation)
        solver_state['hi_costs'] = solver_state['hi_costs'] + torch.nn.ReLU(inplace=True)(dual_cost_perturbation)

        var_lp_f[:, 2] = var_lp_f[:, 2] + scatter_sum(dual_cost_perturbation, edge_index_var_con[0])

        # Dual iterations
        solver_state = sol_utils.dual_iterations(solvers, solver_state, dist_weights, num_dual_iterations, omega, dual_improvement_slope, grad_dual_itr_max_itr)
        solver_state = sol_utils.distribute_delta(solvers, solver_state)

        new_mm = sol_utils.compute_all_min_marginal_diff(solvers, solver_state)
        edge_rest_lp_f[:, 2] = new_mm
        # Update per BDD solution:
        with torch.no_grad(): #TODO: use black-box backprop?
            new_lb = sol_utils.compute_per_bdd_lower_bound(solvers, solver_state)
            con_lp_f[:, 0] = new_lb # Need backprop through lb ?
            edge_rest_lp_f[:, 0] = sol_utils.compute_per_bdd_solution(solvers, solver_state)
       
        return solver_state, var_lp_f, con_lp_f, edge_rest_lp_f

# class DualFullCoordinateAscent(torch.nn.Module):
#     def __init__(self, num_var_lp_f, num_con_lp_f, num_edge_lp_f, depth, var_dim, con_dim, edge_dim):
#         super(DualFullCoordinateAscent, self).__init__()
#         self.feature_refinement = []
#         for d in range(depth):
#             self.feature_refinement.append(FeatureExtractorLayer(num_var_lp_f, var_dim, var_dim,
#                                                                 num_con_lp_f, con_dim, con_dim,   
#                                                                 num_edge_lp_f + 1, edge_dim, edge_dim))

#         self.feature_refinement = torch.nn.ModuleList(self.feature_refinement)
#         self.distribution_weights_predictor = EdgeUpdater(num_edge_lp_f + 1 + edge_dim, 1, 
#                                             num_var_lp_f + var_dim, 
#                                             num_con_lp_f + con_dim, 
#                                             torch.nn.Identity())
    
#     def forward(self, solvers, var_lp_f, con_lp_f, edge_lp_f, 
#                 var_learned_f, con_learned_f, edge_learned_f, 
#                 edge_index_var_con):

#         new_mm = sol_utils.compute_all_min_marginal_diff(solvers, edge_lp_f)
#         edge_lp_f_with_mm = torch.cat((solver_state['lo_costs'], solver_state['hi_costs'], solver_state['def_mm'], edge_rest_lp_f, new_mm), 1)
#         for d in range(len(self.feature_refinement)):
#             var_learned_f, con_learned_f, edge_learned_f = self.feature_refinement[d](
#                     var_learned_f, var_lp_f, con_learned_f, con_lp_f, edge_learned_f, edge_lp_f_with_mm, edge_index_var_con
#                 )
#         # Possibly use some normalization here e.g. layer norm.
#         distribution_weights = self.distribution_weights_predictor(
#             torch.cat((var_learned_f, var_lp_f), 1), torch.cat((con_learned_f, con_lp_f), 1),
#             torch.cat((edge_learned_f, edge_lp_f_with_mm), 1), edge_index_var_con) # Restrict distribution weights to [-eps, \inf] ?

#         # Make distribution weights sum to 1 for each primal variable to ensure dual feasibility.
#         distribution_weights = sol_utils.normalize_distribution_weights(distribution_weights, batch.edge_index_var_con) 
        
#         var_indices = batch.edge_index_var_con[0, :]
#         net_delta = scatter_sum(new_mm, var_indices)[var_indices] * distribution_weights - new_mm

#         # Update costs:
#         edge_lp_f[:, 0] = edge_lp_f[:, 0] + torch.relu(-net_delta) # lo cost
#         edge_lp_f[:, 1] = edge_lp_f[:, 1] - torch.relu(net_delta)  # hi cost

#         # Update per BDD solution:
#         with torch.no_grad(): #TODO: use black-box backprop?
#             edge_lp_f[:, 3] = sol_utils.compute_per_bdd_solution(solvers, edge_lp_f)

#         # Update lower bound for each BDD:
#         con_lp_f[:, 0] = sol_utils.compute_per_bdd_lower_bound(solvers, edge_lp_f)
#         return edge_lp_f, con_lp_f