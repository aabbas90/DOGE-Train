import torch, os, time
import model.solver_utils as sol_utils
from torch_scatter.scatter import scatter_sum, scatter_mean
from metrics.dual_metrics import DualMetrics 
from pytorch_lightning.core.lightning import LightningModule
from typing import List, Set, Dict, Tuple, Optional

class DualActMax(LightningModule):
    def __init__(self, 
                num_test_rounds: int,
                num_dual_iter_test: int,
                dual_improvement_slope_test: float,
                grad_dual_itr_max_itr: int,
                lr: float,
                omega_initial: float,
                use_rel_gap_loss: bool,
                var_lp_features: List[str],
                con_lp_features: List[str],
                edge_lp_features: List[str],               
                var_lp_features_init: List[str],
                con_lp_features_init: List[str],
                edge_lp_features_init: List[str],
                log_every_n_steps: int = 20,
                test_datanames: Optional[List[str]] = None
                ):
        super(DualActMax, self).__init__()
        self.save_hyperparameters(
                'num_test_rounds',
                'num_dual_iter_test', 
                'dual_improvement_slope_test',
                'grad_dual_itr_max_itr', 
                'lr',
                'omega_initial',
                'use_rel_gap_loss',
                'var_lp_features',
                'con_lp_features',
                'edge_lp_features',
                'var_lp_features_init',
                'con_lp_features_init',
                'edge_lp_features_init',
                'test_datanames')

        self.test_datanames = test_datanames
        self.log_every_n_steps = log_every_n_steps

        self.eval_metrics_test = torch.nn.ModuleDict()
        for data_name in test_datanames:
            self.eval_metrics_test[data_name] = DualMetrics(num_test_rounds, num_dual_iter_test)

    @classmethod
    def from_config(cls, cfg, test_datanames, num_dual_iter_test, num_test_rounds, dual_improvement_slope_test):
        return cls(
            num_test_rounds = num_test_rounds,
            var_lp_features = cfg.MODEL.VAR_LP_FEATURES,
            con_lp_features = cfg.MODEL.CON_LP_FEATURES,
            edge_lp_features = cfg.MODEL.EDGE_LP_FEATURES,
            var_lp_features_init = cfg.MODEL.VAR_LP_FEATURES_INIT,
            con_lp_features_init = cfg.MODEL.CON_LP_FEATURES_INIT,
            edge_lp_features_init = cfg.MODEL.EDGE_LP_FEATURES_INIT,
            use_rel_gap_loss = cfg.TRAIN.USE_RELATIVE_GAP_LOSS,
            num_dual_iter_test = num_dual_iter_test,
            dual_improvement_slope_test = dual_improvement_slope_test,
            grad_dual_itr_max_itr = cfg.TRAIN.GRAD_DUAL_ITR_MAX_ITR,
            lr = cfg.TRAIN.BASE_LR,
            omega_initial = cfg.MODEL.OMEGA_INITIAL,
            log_every_n_steps = cfg.LOG_EVERY,
            test_datanames = test_datanames)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        gt_type = None # 'lp_stats' TODOAA
        batch, dist_weights = sol_utils.init_solver_and_get_states(batch, device, gt_type,
                    self.hparams.var_lp_features_init, self.hparams.con_lp_features_init, self.hparams.edge_lp_features_init, 0, 1.0, self.hparams.omega_initial, 
                    distribute_delta = True)
        batch.initial_lb_per_instance = scatter_sum(batch.con_lp_f[:, self.hparams.con_lp_features.index('lb')], batch.batch_index_con)
        batch.edge_rest_lp_f[:, self.hparams.edge_lp_features.index('omega')] = torch.zeros_like(dist_weights)
        
        if batch.gt_info['lp_stats']['obj'][0] is not None:
            batch.gt_obj_normalized = (batch.gt_info['lp_stats']['obj'] - batch.obj_offset) * batch.obj_multiplier
            batch.gt_obj_normalized = batch.gt_obj_normalized.to(device)
            for (b, fp) in enumerate(batch.file_path):
                diff = batch.gt_obj_normalized[b] - batch.initial_lb_per_instance[b]
                try:
                    assert diff > 1e-6, f"lb difference for file: {fp} = {diff} < 1e-6."
                except:
                    breakpoint()
        return batch

    def log_metrics_test(self, metrics_calculator, prefix, suffix):
        assert('test' in prefix)
        metrics_dict = metrics_calculator.compute()
        print(f'\n{prefix}')
        for metric_name, metric_value in metrics_dict.items():
            print(f'\t {metric_name}_{suffix}: ')
            prev_value = None
            for name_itr, value in metric_value.items():
                # self.logger.experiment.add_scalar(f'{mode}/{metric_name}{suffix}', value, global_step = int(itr.replace('itr_', '')))
                name, itr_time = name_itr.split('itr_')
                itr, ctime = itr_time.split('_time_')
                self.logger.experiment.add_scalars(f'{prefix}/{metric_name}_{name}', {suffix: value}, global_step = int(itr), walltime = float(ctime))
                if prev_value is None or prev_value != value:
                    print(f'\t \t {name_itr:30}: {value}')
                prev_value = value
        self.logger.experiment.flush()
        metrics_calculator.reset()

    def log_dist_weights(self, dist_weights, data_name, edge_index_var_con, itr):
        var_indices = edge_index_var_con[0]
        dist_weights_mean = scatter_mean(dist_weights, var_indices)[var_indices]
        dist_weights_variance_per_var = scatter_mean(torch.square(dist_weights - dist_weights_mean), var_indices) / torch.numel(dist_weights)
        self.logger.experiment.add_histogram(f'{data_name}/std_dist_weights', dist_weights_variance_per_var, global_step = itr)

    def log_omega_vector(self, omega_vec, data_name, edge_index_var_con, itr):
        if omega_vec is None:
            return
        var_indices = edge_index_var_con[0]
        omega_vec_mean = scatter_mean(omega_vec, var_indices)
        self.logger.experiment.add_histogram(f'{data_name}/omega_vec_mean', omega_vec_mean, global_step = itr)
        omega_vec_variance_per_var = scatter_mean(torch.square(omega_vec - omega_vec_mean[var_indices]), var_indices) / torch.numel(omega_vec)
        self.logger.experiment.add_histogram(f'{data_name}/std_omega_vec', omega_vec_variance_per_var, global_step = itr)

    def test_epoch_end(self, outputs): # Logs per dataset metrics. Instance level metrics are logged already in test-step(..).
        for data_name in self.test_datanames:
            self.log_metrics_test(self.eval_metrics_test[data_name], f'test_{data_name}', 'learned')
            if self.non_learned_updates_test:
                self.log_metrics_test(self.eval_metrics_test_non_learned[data_name], f'test_{data_name}', 'non_learned')

    def dual_loss_lb(self, lb_after_dist, batch_index_con, initial_lb_per_instance = None, gt_obj_normalized = None):
        # Larger ILPs should have more impact on loss. 
        if not self.hparams.use_rel_gap_loss:
            return -lb_after_dist.sum()
        else:
            if lb_after_dist.requires_grad:
                assert gt_obj_normalized is not None
            elif gt_obj_normalized is None:
                return None

            numer = gt_obj_normalized - scatter_sum(lb_after_dist, batch_index_con)
            denom = gt_obj_normalized - initial_lb_per_instance
            rel_gap = 100.0 * torch.square(numer / (1e-4 + denom))  # Focus more on larger gaps so taking square.
            return rel_gap.sum()

    def single_dual_round(self, batch, num_dual_iterations, improvement_slope, grad_dual_itr_max_itr, gt_obj_normalized):
        prev_dist_weights = batch.edge_rest_lp_f[:, self.hparams.edge_lp_features.index('dist_weights')]
        prev_omega = batch.edge_rest_lp_f[:, self.hparams.edge_lp_features.index('omega')]

        dist_weights_g = prev_dist_weights.clone().detach()
        omega_g = prev_omega.clone().detach()

        with torch.set_grad_enabled(True):
            dist_weights_g.requires_grad = True
            dist_weights_g.retain_grad()
            omega_g.requires_grad = True
            omega_g.retain_grad()
            
            # Make feasible:
            dist_weights_g_feasible = sol_utils.normalize_distribution_weights_softmax(dist_weights_g, batch.edge_index_var_con)
            omega_g_feasible = torch.sigmoid(omega_g)
            batch.solver_state = sol_utils.dual_iterations(batch.solvers, batch.solver_state, dist_weights_g_feasible, num_dual_iterations, omega_g_feasible, 
                                                    improvement_slope, grad_dual_itr_max_itr, 
                                                    self.logger.experiment, batch.file_path, self.global_step)
            
            batch.solver_state = sol_utils.distribute_delta(batch.solvers, batch.solver_state)
            lb_after_dist = sol_utils.compute_per_bdd_lower_bound(batch.solvers, batch.solver_state)
            current_loss = 1000.0 * self.dual_loss_lb(lb_after_dist, batch.batch_index_con, batch.initial_lb_per_instance, gt_obj_normalized)
            current_loss.backward()

            prev_dist_weights = prev_dist_weights - self.hparams.lr * dist_weights_g.grad
            prev_omega = prev_omega - self.hparams.lr * omega_g.grad
            batch.solver_state['lo_costs'] = batch.solver_state['lo_costs'].detach()
            batch.solver_state['hi_costs'] = batch.solver_state['hi_costs'].detach()
            batch.solver_state['def_mm'] = batch.solver_state['def_mm'].detach()

        batch.edge_rest_lp_f[:, self.hparams.edge_lp_features.index('dist_weights')] = prev_dist_weights.detach()
        batch.edge_rest_lp_f[:, self.hparams.edge_lp_features.index('omega')] = prev_omega.detach()
        return batch, lb_after_dist

    def dual_rounds(self, batch, num_rounds, num_dual_iterations, improvement_slope, grad_dual_itr_max_itr, is_training = False, instance_log_name = None, non_learned_updates = False):
        loss = 0
        logs = [{'r' : 0, 'lb_per_instance': scatter_sum(batch.con_lp_f[:, self.hparams.con_lp_features.index('lb')], batch.batch_index_con), 't': time.time()}]
        gt_obj_normalized = None
        if 'gt_obj_normalized' in batch.keys:
            gt_obj_normalized = batch.gt_obj_normalized
        current_loss = self.dual_loss_lb(batch.con_lp_f[:, self.hparams.con_lp_features.index('lb')], batch.batch_index_con, batch.initial_lb_per_instance, gt_obj_normalized)
        if current_loss is not None:
            logs[-1]['loss'] = current_loss.detach()
        for r in range(num_rounds):
            with torch.set_grad_enabled(False):
                batch, lb_after_dist = self.single_dual_round(batch, num_dual_iterations, improvement_slope, grad_dual_itr_max_itr, gt_obj_normalized)
                # if r > num_rounds / 2:
                #     grad_dual_itr_max_itr = 0
                if instance_log_name is not None:
                    self.log_dist_weights(batch.edge_rest_lp_f[:, self.hparams.edge_lp_features.index('dist_weights')], instance_log_name, batch.edge_index_var_con, (r + 1) * num_dual_iterations)
                    self.log_omega_vector(batch.edge_rest_lp_f[:, self.hparams.edge_lp_features.index('omega')], instance_log_name, batch.edge_index_var_con, (r + 1) * num_dual_iterations)

                current_loss = self.dual_loss_lb(lb_after_dist, batch.batch_index_con, batch.initial_lb_per_instance, gt_obj_normalized)
                logs.append({'r' : r + 1, 'lb_per_instance': scatter_sum(lb_after_dist, batch.batch_index_con), 't': time.time()})
                if current_loss is not None:
                    logs[-1]['loss'] = current_loss.detach()
                    loss = loss + current_loss
        return loss, batch, logs

    def test_step(self, batch, batch_idx, dataset_idx = 0):
        assert len(batch.file_path) == 1, 'batch size 1 required for testing.'
        instance_name = os.path.basename(batch.file_path[0])
        data_name = self.test_datanames[dataset_idx]
        instance_log_name = f'test_{data_name}/{instance_name}'
        loss, batch, logs = self.dual_rounds(batch, self.hparams.num_test_rounds, self.hparams.num_dual_iter_test, self.hparams.dual_improvement_slope_test, self.hparams.grad_dual_itr_max_itr, is_training = False, instance_log_name = instance_log_name, non_learned_updates = False)
        instance_level_metrics = DualMetrics(self.hparams.num_test_rounds, self.hparams.num_dual_iter_test).to(batch.edge_index_var_con.device)
        instance_level_metrics.update(batch, logs)
        self.log_metrics_test(instance_level_metrics, f'test_{data_name}_{instance_name}', 'learned')
        self.eval_metrics_test[data_name].update(batch, logs)
        return loss