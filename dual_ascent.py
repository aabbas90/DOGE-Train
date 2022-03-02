import torch
from torch_scatter.scatter import scatter_add
from pytorch_lightning.core.lightning import LightningModule
from typing import List, Set, Dict, Tuple, Optional
from torch.optim import Adam
import logging
import numpy as np
from model.model import FeatureExtractor, DualDistWeightsBlock
import model.solver_utils as sol_utils
from metrics.dual_metrics import DualMetrics 

class DualAscentBDD(LightningModule):
    def __init__(self, 
                num_train_rounds: int,
                num_test_rounds: int,
                num_dual_iter_train: int,
                num_dual_iter_test: int,
                dual_improvement_slope_train: float,
                dual_improvement_slope_test: float,
                grad_dual_itr_max_itr: int,
                lr: float,
                loss_discount_factor:float,
                loss_margin: float,
                omega_initial: float,
                var_lp_features: List[str],  # obj, deg
                con_lp_features: List[str],  # BDD lb, rhs, con type, degree
                edge_lp_features: List[str],  # lo cost, hi cost, mm diff, bdd sol, con coeff, prev dist weights
                num_learned_var_f: int, 
                num_learned_con_f: int, 
                num_learned_edge_f: int,
                feature_extractor_depth: int,
                dual_predictor_depth: int,
                optimizer_name: str,
                datasets: List[str],
                start_episodic_training_after_epoch: int,
                test_fraction: List[int],
                log_every_n_steps: int = 20,
                test_datanames: Optional[List[str]] = None,
                test_uses_full_instances = False,
                non_learned_updates = False
                ):
        super(DualAscentBDD, self).__init__()
        self.save_hyperparameters(
                'num_train_rounds',
                'num_test_rounds',
                'num_dual_iter_train',
                'num_dual_iter_test', 
                'dual_improvement_slope_train',
                'dual_improvement_slope_test',
                'grad_dual_itr_max_itr', 
                'lr',
                'omega_initial',
                'loss_discount_factor',
                'loss_margin',
                'var_lp_features',
                'con_lp_features',
                'edge_lp_features',
                'num_learned_var_f', 
                'num_learned_con_f', 
                'num_learned_edge_f',
                'feature_extractor_depth', 
                'dual_predictor_depth', 
                'optimizer_name',
                'start_episodic_training_after_epoch',
                'datasets',
                'test_fraction',
                'test_datanames',
                'non_learned_updates')

        self.lp_feature_extractor = FeatureExtractor(
                        num_var_lp_f = len(var_lp_features), out_var_dim = num_learned_var_f, 
                        num_con_lp_f = len(con_lp_features), out_con_dim = num_learned_con_f,
                        num_edge_lp_f = len(edge_lp_features), out_edge_dim = num_learned_edge_f,
                        depth = feature_extractor_depth)

        self.dual_block = DualDistWeightsBlock(
                        num_var_lp_f = len(var_lp_features),
                        num_con_lp_f = len(con_lp_features), 
                        num_edge_lp_f = len(edge_lp_features),
                        depth = dual_predictor_depth,
                        var_dim = num_learned_var_f, 
                        con_dim = num_learned_con_f,
                        edge_dim = num_learned_edge_f)

        self.test_datanames = test_datanames
        self.console_logger = logging.getLogger('lightning')
        self.log_every_n_steps = log_every_n_steps
        self.test_uses_full_instances = test_uses_full_instances
        self.train_metrics = DualMetrics(num_train_rounds, num_dual_iter_train)
        self.eval_metrics = torch.nn.ModuleDict()
        self.non_learned_updates_test = non_learned_updates
        for data_name in test_datanames:
            self.eval_metrics[data_name] = DualMetrics(num_test_rounds, num_dual_iter_test)

    @classmethod
    def from_config(cls, cfg, test_datanames, test_uses_full_instances, num_test_rounds, num_dual_iter_test, dual_improvement_slope_test):
        return cls(
            num_train_rounds = cfg.TRAIN.NUM_ROUNDS,
            num_test_rounds = num_test_rounds,
            var_lp_features = cfg.MODEL.VAR_LP_FEATURES,
            con_lp_features = cfg.MODEL.CON_LP_FEATURES,
            edge_lp_features = cfg.MODEL.EDGE_LP_FEATURES,
            num_dual_iter_train = cfg.TRAIN.NUM_DUAL_ITERATIONS,
            num_dual_iter_test = num_dual_iter_test,
            dual_improvement_slope_train = cfg.TRAIN.DUAL_IMPROVEMENT_SLOPE,
            dual_improvement_slope_test = dual_improvement_slope_test,
            grad_dual_itr_max_itr = cfg.TRAIN.GRAD_DUAL_ITR_MAX_ITR,
            lr = cfg.TRAIN.BASE_LR,
            loss_discount_factor = cfg.TRAIN.LOSS_DISCOUNT_FACTOR,
            loss_margin = cfg.TRAIN.LOSS_MARGIN,
            omega_initial = cfg.MODEL.OMEGA_INITIAL,
            num_learned_var_f = cfg.MODEL.VAR_FEATURE_DIM, 
            num_learned_con_f = cfg.MODEL.CON_FEATURE_DIM,
            num_learned_edge_f = cfg.MODEL.EDGE_FEATURE_DIM,
            feature_extractor_depth = cfg.MODEL.FEATURE_EXTRACTOR_DEPTH,
            dual_predictor_depth = cfg.MODEL.DUAL_PRED_DEPTH,
            start_episodic_training_after_epoch = cfg.TRAIN.START_EPISODIC_TRAINING_AFTER_EPOCH,
            optimizer_name = cfg.TRAIN.OPTIMIZER,
            datasets = cfg.DATA.DATASETS,
            test_fraction = cfg.DATA.TEST_FRACTION,
            log_every_n_steps = cfg.LOG_EVERY,
            test_datanames = test_datanames,
            test_uses_full_instances = test_uses_full_instances)

    def configure_optimizers(self):
        if self.hparams.optimizer_name == 'Adam':
            return Adam(self.parameters(), lr=self.hparams.lr)
        else:
            raise ValueError(f'Optimizer {self.hparams.optimizer_name} not exposed.')

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch.edge_index_var_con = batch.edge_index_var_con.to(device)
        batch.omega = torch.tensor([self.hparams.omega_initial], device = device)
        solvers, solver_state, per_bdd_sol, per_bdd_lb, dist_weights, valid_edge_mask, gt_sol_edge, gt_sol_var = sol_utils.init_solver_and_get_states(batch, device, 'lp_stats', 0, 1.0, batch.omega)
        batch.valid_edge_mask = valid_edge_mask
        batch.gt_sol_edge = gt_sol_edge
        batch.gt_sol_var = gt_sol_var

        # Variable LP features:
        var_degree = scatter_add(torch.ones((batch.num_edges), device=device), batch.edge_index_var_con[0])
        var_degree[torch.cumsum(batch.num_vars, 0) - 1] = 0 # Terminal nodes, not corresponding to any primal variable.
        batch.var_lp_f = torch.stack((batch.objective.to(device), var_degree), 1) # Obj, Deg.
        batch.objective = None

        # Constraint LP features:
        con_degree = scatter_add(torch.ones((batch.num_edges), device=device), batch.edge_index_var_con[1])
        batch.con_lp_f = torch.stack((per_bdd_lb, batch.rhs_vector.to(device), batch.con_type.to(device), con_degree), 1) # BDD lb, rhs, con type, degree
        batch.rhs_vector = None
        batch.con_type = None

        # Edge LP features:
        batch.edge_rest_lp_f = torch.stack((per_bdd_sol, batch.con_coeff.to(device), dist_weights), 1) #TODOAA: Compute moving average of previous dist weights as features?
        batch.solver_state = solver_state
        
        batch.solvers = solvers
        return batch

    def compute_training_start_round(self):
        current_start_epoch = self.current_epoch - self.hparams.start_episodic_training_after_epoch
        max_training_epoch_mod = 1 + (self.trainer.max_epochs // 10) # Divides into ten journeys.
        fraction = float(current_start_epoch) / max_training_epoch_mod
        if fraction < 0:
            return 0
        fraction = fraction % 1
        fraction = fraction * fraction
        mean_start_step = fraction * (self.hparams.num_train_rounds)
        proposed_start_step = np.round(np.random.normal(mean_start_step, 3)).astype(np.int32).item(0)
        return max(min(proposed_start_step, self.hparams.num_train_rounds - 1), 0)

    def log_metrics(self, metrics_calculator, mode):
        metrics_dict = metrics_calculator.compute()
        for metric_name, metric_value in metrics_dict.items():
            self.logger.experiment.add_scalars(f'{mode}/{metric_name}', metric_value, global_step = self.global_step)
        self.logger.experiment.flush()
        metrics_calculator.reset()

    def log_metrics_test(self, metrics_calculator, mode):
        assert('test' in mode)
        suffix = 'non_learned' if self.non_learned_updates_test else 'learned'
        metrics_dict = metrics_calculator.compute()
        self.console_logger.critical(f'\n{mode}')
        for metric_name, metric_value in metrics_dict.items():
            self.console_logger.critical(f'\t {metric_name}{suffix}: ')
            prev_value = None
            for itr, value in metric_value.items():
                # self.logger.experiment.add_scalar(f'{mode}/{metric_name}{suffix}', value, global_step = int(itr.replace('itr_', '')))
                self.logger.experiment.add_scalars(f'{mode}/{metric_name}', {suffix: value}, global_step = int(itr.replace('itr_', '')))
                if prev_value is None or prev_value != value:
                    self.console_logger.critical(f'\t \t {itr}: {value}')
                prev_value = value
        self.logger.experiment.flush()
        metrics_calculator.reset()

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        data_name = self.test_datanames[dataloader_idx]
        if self.test_uses_full_instances:
            self.log_metrics_test(self.eval_metrics[data_name], f'test_{data_name}')

    def training_epoch_end(self, outputs):
        self.log_metrics(self.train_metrics, 'train')

    def validation_epoch_end(self, outputs):
        for data_name in self.test_datanames:
            self.log_metrics(self.eval_metrics[data_name], f'val_{data_name}')

    def test_epoch_end(self, outputs):
        if self.test_uses_full_instances:
            return # Already logged.
        for data_name in self.test_datanames:
            self.log_metrics_test(self.eval_metrics[data_name], f'test_{data_name}')

    def try_concat_gt_edge_solution(self, batch, is_training):
        edge_sol = None
        for file_path, current_sol in zip(batch.file_path, batch.gt_sol_edge):
            if current_sol is None:
                assert current_sol is not None or not is_training, f'gt solution should be known for files: {file_path}'
                return edge_sol
        return torch.cat(batch.gt_sol_edge, 0)

    def dual_loss(self, mm_pred, gt_lp_sol_edge, valid_edge_mask):
        if gt_lp_sol_edge is None:
            return None

         # Gather the solution w.r.t all valid edges
        mm_pred_valid = torch.nn.Tanh()(10.0 * mm_pred[valid_edge_mask]) 
        gt_hi = gt_lp_sol_edge >= 1.0 - 1e-8 # Here mm_diff should ideally be < 0.
        gt_lo = gt_lp_sol_edge <= 0.0 + 1e-8 # Here mm_diff should ideally be > 0.
        gt_frac = torch.logical_and(~gt_hi, ~gt_lo) # Here mm_diff should ideally be 0.
        mm_hi_pred = mm_pred_valid[gt_hi] + self.hparams.loss_margin
        loss_hi = torch.sum(torch.relu(mm_hi_pred))
        mm_lo_pred = -(mm_pred_valid[gt_lo] - self.hparams.loss_margin)
        loss_lo = torch.sum(torch.relu(mm_lo_pred))
        loss_frac = torch.sum(mm_pred_valid[gt_frac])
        return loss_hi + loss_lo + loss_frac

    def single_dual_round(self, batch, num_dual_iterations, improvement_slope, grad_dual_itr_max_itr):
        batch.solver_state, batch.var_lp_f, batch.con_lp_f, batch.edge_rest_lp_f = self.dual_block(
                                                                batch.solvers, batch.var_lp_f, batch.con_lp_f, 
                                                                batch.solver_state, batch.edge_rest_lp_f, 
                                                                batch.var_learned_f, batch.con_learned_f, batch.edge_learned_f, 
                                                                batch.omega, batch.edge_index_var_con,
                                                                num_dual_iterations, grad_dual_itr_max_itr, improvement_slope, batch.valid_edge_mask)
        batch.solver_state['def_mm'][~batch.valid_edge_mask] = 0 # Locations of terminal nodes can contain nans.
        try:
            assert(torch.all(torch.isfinite(batch.solver_state['def_mm'])))
        except:
            breakpoint()
        return batch

    def dual_rounds(self, batch, num_rounds, num_dual_iterations, improvement_slope, grad_dual_itr_max_itr, is_training = False, data_name = '', non_learned_updates = False):
        if not non_learned_updates:
            batch.var_learned_f, batch.con_learned_f, batch.edge_learned_f = self.lp_feature_extractor(batch.var_lp_f, batch.con_lp_f, batch.solver_state, batch.edge_rest_lp_f, batch.edge_index_var_con)
        loss = 0
        gt_lp_sol_edge = self.try_concat_gt_edge_solution(batch, is_training)
        for r in range(num_rounds):
            with torch.set_grad_enabled(r == num_rounds - 1 and is_training):
                if not non_learned_updates:
                    batch = self.single_dual_round(batch, num_dual_iterations, improvement_slope, grad_dual_itr_max_itr)
                else:
                    with torch.no_grad():
                        batch = sol_utils.non_learned_updates(batch, num_dual_iterations, improvement_slope = 0.0, omega = batch.omega.item())
                # Apply loss on predicted all mm differences with decay weight.
                solver_state = sol_utils.distribute_delta(batch.solvers, batch.solver_state)
                all_mm_diff = sol_utils.compute_all_min_marginal_diff(batch.solvers, solver_state)
                current_loss = self.dual_loss(all_mm_diff, gt_lp_sol_edge, batch.valid_edge_mask)
                #current_loss = -batch.con_lp_f[:, 0].sum()
                if current_loss is not None:
                    loss = loss + torch.pow(torch.tensor(self.hparams.loss_discount_factor), num_rounds - r - 1) * current_loss

                metric_to_log = self.train_metrics if is_training else self.eval_metrics[data_name]
                with torch.no_grad():
                    solver_state = sol_utils.distribute_delta(batch.solvers, batch.solver_state)
                    all_mm_diff = sol_utils.compute_all_min_marginal_diff(batch.solvers, solver_state)
                    metric_to_log.update(current_loss, all_mm_diff, batch.edge_index_var_con, batch.num_vars,
                                        batch.con_lp_f[:, 0], batch.num_cons, batch.gt_info['lp_stats']['obj'], 
                                        r, batch.obj_multiplier, batch.obj_offset)

        return loss

    def training_step(self, batch, batch_idx):
        num_rounds = self.compute_training_start_round() + 1
        loss = self.dual_rounds(batch, num_rounds, self.hparams.num_dual_iter_train, 
                self.hparams.dual_improvement_slope_train, self.hparams.grad_dual_itr_max_itr, is_training = True)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx = 0):
        loss = self.dual_rounds(batch, self.hparams.num_test_rounds, self.hparams.num_dual_iter_test, 
                    self.hparams.dual_improvement_slope_test, 0, is_training = False, data_name = self.test_datanames[dataset_idx])
        return loss

    def test_step(self, batch, batch_idx, dataset_idx = 0):
        loss = self.dual_rounds(batch, self.hparams.num_test_rounds, self.hparams.num_dual_iter_test, 
                    self.hparams.dual_improvement_slope_test, 0, is_training = False, data_name = self.test_datanames[dataset_idx], 
                    non_learned_updates = self.non_learned_updates_test)
        return loss