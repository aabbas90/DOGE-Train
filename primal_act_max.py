import torch, os
import model.solver_utils as sol_utils
from torch_scatter.scatter import scatter_sum, scatter_mean
from metrics.primal_metrics import PrimalMetrics 
from pytorch_lightning.core.lightning import LightningModule
from typing import List, Set, Dict, Tuple, Optional

class PrimalActMax(LightningModule):
    def __init__(self, 
                num_test_rounds: int,
                num_dual_iter_test: int,
                dual_improvement_slope_test: float,
                grad_dual_itr_max_itr: int,
                lr: float,
                min_perturbation: float,
                omega: float,
                var_lp_features: List[str],
                con_lp_features: List[str],
                edge_lp_features: List[str],               
                var_lp_features_init: List[str],
                con_lp_features_init: List[str],
                edge_lp_features_init: List[str],
                log_every_n_steps: int = 20,
                test_datanames: Optional[List[str]] = None
                ):
        super(PrimalActMax, self).__init__()
        self.save_hyperparameters(
                'num_test_rounds',
                'num_dual_iter_test', 
                'dual_improvement_slope_test',
                'grad_dual_itr_max_itr', 
                'lr',
                'min_perturbation',
                'omega',
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
            self.eval_metrics_test[data_name] = PrimalMetrics(num_test_rounds, self.hparams.con_lp_features)

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
            num_dual_iter_test = num_dual_iter_test,
            dual_improvement_slope_test = dual_improvement_slope_test,
            grad_dual_itr_max_itr = cfg.TRAIN.GRAD_DUAL_ITR_MAX_ITR,
            lr = cfg.TRAIN.BASE_LR,
            min_perturbation = cfg.TRAIN.MIN_PERTURBATION,
            omega = cfg.MODEL.OMEGA,
            log_every_n_steps = cfg.LOG_EVERY,
            test_datanames = test_datanames)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch, dist_weights = sol_utils.init_solver_and_get_states(batch, device, 'ilp_stats', 
                                self.hparams.var_lp_features_init, self.hparams.con_lp_features_init, self.hparams.edge_lp_features_init,
                                0, 1.0, self.hparams.omega, distribute_delta=True)
 
         # Normalize original costs as they will be sent to GNN:
        var_costs = batch.var_lp_f[:, self.hparams.var_lp_features.index('orig_obj')]
        lb_per_bdd = batch.con_lp_f[:, self.hparams.con_lp_features.index('orig_lb')]
        mm_diff = batch.edge_rest_lp_f[:, self.hparams.edge_lp_features.index('orig_mm_diff')]
        var_costs, lb_per_bdd, mm_diff, batch.var_cost_mean, batch.var_cost_std = sol_utils.normalize_costs_var(
            var_costs, lb_per_bdd, mm_diff, batch.num_cons, batch.batch_index_var, batch.batch_index_con, batch.batch_index_edge)
        batch.var_lp_f[:, self.hparams.var_lp_features.index('orig_obj')] = var_costs
        batch.con_lp_f[:, self.hparams.con_lp_features.index('orig_lb')] = lb_per_bdd
        batch.edge_rest_lp_f[:, self.hparams.edge_lp_features.index('orig_mm_diff')] = mm_diff

        batch.dist_weights = dist_weights # Isotropic weights.
        batch.primal_perturbation_var = torch.zeros_like(var_costs)

        return batch

    def log_metrics_test(self, metrics_calculator, prefix, suffix):
        assert('test' in prefix)
        metrics_dict = metrics_calculator.compute()
        print(f'\n{prefix}')
        for metric_name, metric_value_per_round_dict in metrics_dict.items():
            print(f'\t {metric_name}_{suffix}: ')
            prev_value = None
            for round, value in metric_value_per_round_dict.items():
                self.logger.experiment.add_scalar(f'{prefix}/{metric_name}_{suffix}', value, global_step = int(round.replace('round_', '')))
                if prev_value is None or prev_value != value:
                    print(f'\t \t {round}: {value}')
                prev_value = value
        self.logger.experiment.flush()
        metrics_calculator.reset()

    def test_epoch_end(self, outputs):
        for data_name in self.test_datanames:
            self.log_metrics_test(self.eval_metrics_test[data_name], f'test_{data_name}', 'act_max')

    def loss_on_lb_increase(self, con_lp_f, batch_index_con, orig_var_cost_mean, orig_var_cost_std):
        # Larger problems should have more lb increase so taking sum directly:
        prev_lb_per_instance = scatter_sum(con_lp_f[:, self.hparams.con_lp_features.index('prev_lb')], batch_index_con)
        orig_lb_per_instance = scatter_sum(con_lp_f[:, self.hparams.con_lp_features.index('orig_lb')], batch_index_con) * orig_var_cost_std + orig_var_cost_mean
        return (prev_lb_per_instance - orig_lb_per_instance).mean()  

    def single_primal_round(self, batch, primal_perturbation_var, dist_weights):
        assert(torch.all(torch.isfinite(primal_perturbation_var)))
        var_indices = batch.edge_index_var_con[0]
        primal_perturbation_var_lo = -primal_perturbation_var + self.hparams.min_perturbation * 0.5
        primal_perturbation_var_hi = primal_perturbation_var + self.hparams.min_perturbation * 0.5

        p_lo_cost_var = torch.relu(primal_perturbation_var_lo)
        p_hi_cost_var = torch.relu(primal_perturbation_var_hi)
        
        p_lo_cost = p_lo_cost_var[var_indices] * dist_weights
        p_hi_cost = p_hi_cost_var[var_indices] * dist_weights

        pert_solver_state = {
                            'lo_costs': batch.solver_state['lo_costs'] + p_lo_cost, 
                            'hi_costs': batch.solver_state['hi_costs'] + p_hi_cost,
                            'def_mm': batch.solver_state['def_mm']
                            }
        
        # Dual iterations
        pert_solver_state = sol_utils.dual_iterations(batch.solvers, pert_solver_state, dist_weights, self.hparams.num_dual_iter_test, 
                                                    batch.omega, self.hparams.dual_improvement_slope_test, self.hparams.grad_dual_itr_max_itr)
        pert_solver_state = sol_utils.distribute_delta(batch.solvers, pert_solver_state)

        batch.edge_rest_lp_f[:, self.hparams.edge_lp_features.index('prev_mm_diff')] = sol_utils.compute_all_min_marginal_diff(batch.solvers, pert_solver_state)
        batch.con_lp_f[:, self.hparams.con_lp_features.index('prev_lb')] = sol_utils.compute_per_bdd_lower_bound(batch.solvers, pert_solver_state) # Update perturbed lower bound.
        prev_var_costs = scatter_sum(pert_solver_state['hi_costs'] - pert_solver_state['lo_costs'], var_indices) # Convert to variable costs.
        batch.var_lp_f[:, self.hparams.var_lp_features.index('prev_obj')] = prev_var_costs

        # Update per BDD solution:
        with torch.no_grad():
            batch.edge_rest_lp_f[:, self.hparams.edge_lp_features.index('prev_sol')] = sol_utils.compute_per_bdd_solution(batch.solvers, pert_solver_state)
        return batch.var_lp_f, batch.con_lp_f, batch.edge_rest_lp_f

    def primal_rounds(self, batch):
        logs = []
        primal_perturbation_var = batch.primal_perturbation_var.clone().detach()
        dist_weights = batch.dist_weights.clone().detach()
        for r in range(self.hparams.num_test_rounds):
            with torch.set_grad_enabled(True):
                primal_perturbation_var_g = primal_perturbation_var.clone().detach()
                primal_perturbation_var_g.requires_grad = True
                primal_perturbation_var_g.retain_grad()

                # dist_weights_g = dist_weights.clone().detach()
                # dist_weights_g.requires_grad = True
                # dist_weights_g.retain_grad()
                # dist_weights_gp = sol_utils.normalize_distribution_weights_softmax(dist_weights_g, batch.edge_index_var_con)

                batch.var_lp_f, batch.con_lp_f, batch.edge_rest_lp_f = self.single_primal_round(batch, primal_perturbation_var_g, dist_weights)
                all_mm_diff = batch.edge_rest_lp_f[:, self.hparams.edge_lp_features.index('prev_mm_diff')].clone().detach()
                pert_lb = batch.con_lp_f[:, self.hparams.con_lp_features.index('prev_lb')].clone().detach()
                current_loss = self.loss_on_lb_increase(batch.con_lp_f, batch.batch_index_con, batch.var_cost_mean, batch.var_cost_std)
                logs.append({'r' : r, 'all_mm_diff': all_mm_diff.to('cpu'), 'prev_lb': pert_lb.to('cpu')})
                if current_loss is not None:
                    logs[-1]['loss'] = current_loss.detach()

                current_loss.backward()
                primal_perturbation_var = primal_perturbation_var - self.hparams.lr *  primal_perturbation_var_g.grad
                # dist_weights = dist_weights - self.hparams.lr *  dist_weights_g.grad

                batch.var_lp_f = batch.var_lp_f.detach()
                batch.con_lp_f = batch.con_lp_f.detach()
                batch.edge_rest_lp_f = batch.edge_rest_lp_f.detach()
                print(f'Round: {r}: Loss: {current_loss.item():.3f}')
            
            hi_assignments = all_mm_diff < -1e-6
            lo_assignments = all_mm_diff > 1e-6
            var_hi = scatter_mean(hi_assignments.to(torch.float32), batch.edge_index_var_con[0]) >= 1.0 - 1e-6
            var_lo = scatter_mean(lo_assignments.to(torch.float32), batch.edge_index_var_con[0]) >= 1.0 - 1e-6
            var_lo[-1] = True # terminal node.
            # if (var_hi + var_lo).min() >= 1.0 - 1e-6: # Solution found
            #     return batch, logs

        return batch, logs

    def test_step(self, batch, batch_idx, dataset_idx = 0):
        assert len(batch.file_path) == 1, 'batch size 1 required for testing.'
        instance_name = os.path.basename(batch.file_path[0])
        data_name = self.test_datanames[dataset_idx]

        batch_updated, logs = self.primal_rounds(batch)
        instance_level_metrics = PrimalMetrics(self.hparams.num_test_rounds, self.hparams.con_lp_features).to(batch.edge_index_var_con.device)
        instance_level_metrics.update(batch_updated, logs)
        self.log_metrics_test(instance_level_metrics, f'test_{data_name}_{instance_name}', 'act_max')
        self.eval_metrics_test[data_name].update(batch_updated, logs)
        return 0