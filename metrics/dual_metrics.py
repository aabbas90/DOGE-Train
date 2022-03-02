import torch
from torchmetrics import Metric
from torch_scatter import scatter_mean

class DualMetrics(Metric):
    def __init__(self, num_rounds, num_dual_iter_per_round, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        default = torch.zeros(num_rounds, dtype=torch.float32)
        self.num_dual_iter_per_round = num_dual_iter_per_round
        self.add_state('loss', default=default, dist_reduce_fx="sum")
        self.add_state('pred_lb_sums', default=default, dist_reduce_fx="sum")
        self.add_state('gt_obj_sums', default=default, dist_reduce_fx="sum")
        self.add_state('num_disagreements', default=default, dist_reduce_fx="sum")
        self.add_state('total_instances', default=default + 1e-7, dist_reduce_fx="sum")
        self.add_state('gt_known', default = torch.ones(1, dtype=torch.bool), dist_reduce_fx="min")
        self.add_state('max_round', default = torch.zeros(1, dtype=torch.int32), dist_reduce_fx="max")

    def update(self, 
            loss, # can be None if GT not known.
            mm_pred_edge: torch.Tensor, 
            edge_index_var_con: torch.Tensor,
            num_vars_per_inst: torch.Tensor, 
            pred_lb_per_bdd,
            num_bdds_per_inst,
            gt_obj_batch, # List of gt objectives for all items in batch. Any of the items can also be None if gt not present.
            round: int,
            obj_multipliers: torch.Tensor,
            obj_offsets: torch.Tensor):
        if loss is not None:
            self.loss[round] += loss
        self.max_round[0] = max(self.max_round[0].item(), round)
        var_indices = edge_index_var_con[0]
        mm_pred_sign = torch.sign(mm_pred_edge)
        variable_mean_sign = scatter_mean(mm_pred_sign, var_indices)
        assignment_lo = variable_mean_sign >= 1.0 - 1e-6
        assignment_hi = variable_mean_sign <= -(1.0 - 1e-6)
        vars_agree = torch.logical_or(assignment_hi, assignment_lo)
        bdd_start = 0
        var_start = 0
        for b, (gt_obj, num_bdds, num_vars) in enumerate(zip(gt_obj_batch, num_bdds_per_inst, num_vars_per_inst)):
            var_end = num_vars + var_start - 1
            current_lb = pred_lb_per_bdd[bdd_start: bdd_start + num_bdds].sum()
            self.pred_lb_sums[round] += (current_lb / obj_multipliers[b]) + obj_offsets[b]
            self.total_instances[round] += 1.0
            if gt_obj is not None:
                self.gt_obj_sums[round] += gt_obj
            else:
                self.gt_known[0] = False
            self.num_disagreements[round] += torch.logical_not(vars_agree[var_start:var_end]).sum() / num_vars

            bdd_start += num_bdds

    def compute(self):
        # compute final result
        merged_results = {}
        num_valid_rounds = self.max_round[0].item() + 1
        if self.gt_known[0].item():
            gt_obj_mean = self.gt_obj_sums / self.total_instances

        pred_lb_mean = self.pred_lb_sums / self.total_instances
        percentage_disagreements = 100.0 * self.num_disagreements / self.total_instances

        merged_results['pred_mean_lb'] = {}
        merged_results['percentage_disagreements'] = {}
        for r in range(num_valid_rounds):
            tag = f'itr_{r * self.num_dual_iter_per_round}'
            merged_results['pred_mean_lb'][tag] = pred_lb_mean[r]
            merged_results['percentage_disagreements'][tag] = percentage_disagreements[r]
            if self.gt_known:
                if not 'loss' in merged_results:
                    merged_results['loss'] = {}
                merged_results['loss'].update({tag: self.loss[r]})
                if not 'gt_mean_lb' in merged_results:
                    merged_results['gt_mean_lb'] = {}
                merged_results['gt_mean_lb'].update({tag:  gt_obj_mean[r]})
        return merged_results