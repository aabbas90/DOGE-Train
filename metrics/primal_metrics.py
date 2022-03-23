import torch
from torchmetrics import Metric
from torch_scatter import scatter_mean, scatter_sum

class PrimalMetrics(Metric):
    def __init__(self, num_rounds, con_lp_f_names, on_baseline = False, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        default = torch.zeros(num_rounds, dtype=torch.float32)
        self.on_baseline = on_baseline
        self.con_lp_f_names = con_lp_f_names
        self.add_state('loss', default=default, dist_reduce_fx="sum")
        self.add_state('pred_objective_sums', default=default, dist_reduce_fx="sum")
        self.add_state('gt_objective_sums', default=default, dist_reduce_fx="sum")
        self.add_state('num_disagreements', default=default, dist_reduce_fx="sum")
        self.add_state('num_solved', default=default + 1e-7, dist_reduce_fx="sum")
        if not on_baseline:
            self.add_state('pred_lb_sums', default=default + 1e-7, dist_reduce_fx="sum")
        self.add_state('total_instances', default=default + 1e-7, dist_reduce_fx="sum")
        self.add_state('gt_known', default = torch.ones(1, dtype=torch.bool), dist_reduce_fx="min")
        self.add_state('max_round', default = torch.zeros(1, dtype=torch.int32), dist_reduce_fx="max")
        self.add_state('initial_lb_sums', default = torch.zeros(1, dtype=torch.float32), dist_reduce_fx="sum")

    def compute_per_instance_lb(self, batch, per_bdd_lb):
        # First get rid of standardization.
        per_instance_lb = scatter_sum(per_bdd_lb, batch.batch_index_con) * batch.var_cost_std + batch.var_cost_mean
        # Account for normalization during BDD conversion.
        per_instance_lb = per_instance_lb / batch.obj_multiplier.to(per_instance_lb.device) + batch.obj_offset.to(per_instance_lb.device)
        return per_instance_lb

    def update(self, batch, logs):
        with torch.no_grad():
            gt_ilp_sol_var = batch.gt_sol_var
            edge_index_var_con = batch.edge_index_var_con
            device = edge_index_var_con.device
            var_indices = edge_index_var_con[0]
            num_vars_per_instance = batch.num_vars
            orig_obj_vector = batch.objective.to(device)
            obj_multipliers = batch.obj_multiplier
            obj_offsets = batch.obj_offset
            initial_per_instance_lb = self.compute_per_instance_lb(batch, batch.con_lp_f[:, self.con_lp_f_names.index('orig_lb')])
            self.initial_lb_sums += initial_per_instance_lb.sum() 

            for logs_round in logs:
                round = logs_round['r']
                if 'loss' in logs_round:
                    self.loss[round] += logs_round['loss']
                self.max_round[0] = max(self.max_round[0].item(), round)
                if not self.on_baseline:
                    per_instance_lb = self.compute_per_instance_lb(batch, logs_round['prev_lb'].to(device))
                    self.pred_lb_sums[round] = self.pred_lb_sums[round] + per_instance_lb.sum()
                mm_pred_edge = logs_round['all_mm_diff'].to(device)
                mm_pred_sign = torch.sign(mm_pred_edge)
                variable_mean_sign = scatter_mean(mm_pred_sign, var_indices)
                assignment_lo = variable_mean_sign >= 1.0 - 1e-4
                assignment_hi = variable_mean_sign <= -(1.0 - 1e-4)
                vars_agree = torch.logical_or(assignment_hi, assignment_lo)
                prev_var_start = 0
                for (b, current_num_vars) in enumerate(num_vars_per_instance):
                    var_end = current_num_vars + prev_var_start - 1
                    is_decodable = torch.all(vars_agree[prev_var_start:var_end])
                    if is_decodable:
                        current_sol = torch.zeros((current_num_vars), device=device)
                        current_sol[assignment_hi[prev_var_start:var_end + 1]] = 1.0
                        current_sol[-1] = 0 # Terminal node.
                        current_sol_cost = torch.sum(orig_obj_vector[prev_var_start:var_end + 1] * current_sol)
                        self.pred_objective_sums[round] += (current_sol_cost / obj_multipliers[b]) + obj_offsets[b]
                        if gt_ilp_sol_var[b] is not None:
                            gt_sol_cost = torch.sum(orig_obj_vector[prev_var_start:var_end + 1] * gt_ilp_sol_var[b])
                            self.gt_objective_sums[round] += (gt_sol_cost / obj_multipliers[b]) + obj_offsets[b]
                        else:
                            self.gt_known[0] = False
                        self.num_solved[round] += 1.0
                    else:
                        self.num_disagreements[round] += torch.logical_not(vars_agree[prev_var_start:var_end]).sum()

                    self.total_instances[round] += 1.0
                    prev_var_start = var_end + 1 # To account for terminal node.

    def compute(self):
        num_valid_rounds = self.max_round[0].item() + 1
        pred_mean_obj_of_solved = self.pred_objective_sums / self.num_solved
        if self.gt_known[0].item():
            gt_mean_obj_of_solved = self.gt_objective_sums / self.num_solved
        
        merged_results = {}
        merged_results['num_disagreements'] = {}
        if not self.on_baseline:
            pred_lb_mean = self.pred_lb_sums / self.total_instances
            # Round 0 contains original lower bound:
            merged_results['perturbed_mean_lb'] = {'round_0': (self.initial_lb_sums[0] / self.total_instances[0]).item()}
        for r in range(num_valid_rounds):
            merged_results['num_disagreements'][f'round_{r+1}'] = self.num_disagreements[r]
            if not self.on_baseline:
                merged_results['perturbed_mean_lb'][f'round_{r+1}'] = pred_lb_mean[r]
            if self.num_solved[r] == self.total_instances[r]:
                if not 'pred_mean_obj' in merged_results:
                    merged_results['pred_mean_obj'] = {}
                merged_results['pred_mean_obj'].update({f'round_{r+1}': pred_mean_obj_of_solved[r]})
                if self.gt_known:
                    if not 'gt_mean_obj' in merged_results:
                        merged_results['gt_mean_obj'] = {}
                    merged_results['gt_mean_obj'].update({f'round_{r+1}': gt_mean_obj_of_solved[r]})

            if self.gt_known:
                if not 'loss' in merged_results:
                    merged_results['loss'] = {}
                merged_results['loss'].update({f'round_{r+1}': self.loss[r]})
        return merged_results