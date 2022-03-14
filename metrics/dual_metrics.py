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
        self.add_state('wall_time', default=torch.zeros(num_rounds, dtype=torch.float64), dist_reduce_fx="max")
        self.add_state('loss', default=default, dist_reduce_fx="sum")
        self.add_state('pred_lb_sums', default=default, dist_reduce_fx="sum")
        self.add_state('gt_obj_sums', default=default, dist_reduce_fx="sum")
        self.add_state('total_instances', default=default + 1e-7, dist_reduce_fx="sum")
        self.add_state('gt_known', default = torch.ones(1, dtype=torch.bool), dist_reduce_fx="min")
        self.add_state('max_round', default = torch.zeros(1, dtype=torch.int32), dist_reduce_fx="max")

    def update(self, batch, logs):
        with torch.no_grad():
            for logs_round in logs:
                round = logs_round['r']
                time = logs_round['t']
                self.wall_time[round] = max(self.wall_time[round].item(), time)
                if 'loss' in logs_round:
                    self.loss[round] += logs_round['loss']
                self.max_round[0] = max(self.max_round[0].item(), round)
                bdd_start = 0
                for b, (gt_obj, num_bdds) in enumerate(zip(batch.gt_info['lp_stats']['obj'], batch.num_cons)):
                    current_lb = logs_round['lb_per_bdd'][bdd_start: bdd_start + num_bdds].sum()
                    self.pred_lb_sums[round] += (current_lb / batch.obj_multiplier[b]) + batch.obj_offset[b]
                    self.total_instances[round] += 1.0
                    if gt_obj is not None:
                        self.gt_obj_sums[round] += gt_obj
                    else:
                        self.gt_known[0] = False
                    bdd_start += num_bdds

    def compute(self):
        # compute final result
        merged_results = {}
        num_valid_rounds = self.max_round[0].item() + 1
        if self.gt_known[0].item():
            gt_obj_mean = self.gt_obj_sums / self.total_instances
        pred_lb_mean = self.pred_lb_sums / self.total_instances
        lower_bounds = {}
        for r in range(num_valid_rounds):
            tag = f'itr_{r * self.num_dual_iter_per_round + self.num_dual_iter_per_round}_time_{self.wall_time[r]}'
            lower_bounds[f'pred_mean_lb_{tag}'] = pred_lb_mean[r]
            if self.gt_known:
                if not 'loss' in merged_results:
                    merged_results['loss'] = {}
                merged_results['loss'].update({tag: self.loss[r]})
                lower_bounds[f'gt_mean_lb_{tag}'] = gt_obj_mean[r]
        merged_results['lower_bounds'] = lower_bounds
        return merged_results