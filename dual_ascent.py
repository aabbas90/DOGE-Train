import torch
from torch_scatter.scatter import scatter_add, scatter_mean
from pytorch_lightning.core.lightning import LightningModule
from typing import List, Set, Dict, Tuple, Optional
from torch.optim import Adam
import logging, os, time
import numpy as np
from model.model import FeatureExtractor, DualDistWeightsBlock, DualFullCoordinateAscent
import model.solver_utils as sol_utils
from metrics.dual_metrics import DualMetrics 

class DualAscentBDD(LightningModule):
    def __init__(self, 
                num_train_rounds: int,
                num_train_rounds_with_grad: int,
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
                start_episodic_training_after_epoch: int,
                val_fraction: List[int],
                use_layer_norm: True,
                skip_connections: Optional[bool] = False,
                predict_omega: Optional[bool] = False,
                full_coordinate_ascent: Optional[bool] = False,
                val_datanames: Optional[List[str]] = None,
                test_datanames: Optional[List[str]] = None,
                non_learned_updates_test = False
                ):
        super(DualAscentBDD, self).__init__()
        self.save_hyperparameters(
                'num_train_rounds',
                'num_train_rounds_with_grad',
                'num_test_rounds',
                'num_dual_iter_train',
                'num_dual_iter_test', 
                'dual_improvement_slope_train',
                'dual_improvement_slope_test',
                'grad_dual_itr_max_itr', 
                'lr',
                'use_layer_norm',
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
                'full_coordinate_ascent',
                'predict_omega',
                'start_episodic_training_after_epoch',
                'val_fraction',
                'val_datanames',
                'test_datanames')

        self.lp_feature_extractor = FeatureExtractor(
                        num_var_lp_f = len(var_lp_features), out_var_dim = num_learned_var_f, 
                        num_con_lp_f = len(con_lp_features), out_con_dim = num_learned_con_f,
                        num_edge_lp_f = len(edge_lp_features), out_edge_dim = num_learned_edge_f,
                        depth = feature_extractor_depth, use_layer_norm=use_layer_norm,
                        skip_connections = skip_connections)

        if not full_coordinate_ascent:
            self.dual_block = DualDistWeightsBlock(
                            num_var_lp_f = len(var_lp_features),
                            num_con_lp_f = len(con_lp_features), 
                            num_edge_lp_f = len(edge_lp_features),
                            depth = dual_predictor_depth,
                            var_dim = num_learned_var_f, 
                            con_dim = num_learned_con_f,
                            edge_dim = num_learned_edge_f,
                            use_layer_norm = use_layer_norm,
                            predict_omega = predict_omega,
                            skip_connections = skip_connections)
        else:
            self.dual_block = DualFullCoordinateAscent(
                            num_var_lp_f = len(var_lp_features),
                            num_con_lp_f = len(con_lp_features), 
                            num_edge_lp_f = len(edge_lp_features),
                            depth = dual_predictor_depth,
                            var_dim = num_learned_var_f, 
                            con_dim = num_learned_con_f,
                            edge_dim = num_learned_edge_f,
                            use_layer_norm = use_layer_norm)

        self.val_datanames = val_datanames
        self.test_datanames = test_datanames
        self.console_logger = logging.getLogger('lightning')
        self.train_metrics = DualMetrics(num_train_rounds, num_dual_iter_train)

        self.eval_metrics_val = torch.nn.ModuleDict()
        self.eval_metrics_val_non_learned = torch.nn.ModuleDict()
        self.non_learned_updates_val = True
        for data_name in val_datanames:
            self.eval_metrics_val[data_name] = DualMetrics(num_test_rounds, num_dual_iter_test)
            self.eval_metrics_val_non_learned[data_name] = DualMetrics(num_test_rounds, num_dual_iter_test)

        self.non_learned_updates_test = non_learned_updates_test
        self.eval_metrics_test = torch.nn.ModuleDict()
        self.eval_metrics_test_non_learned = torch.nn.ModuleDict()
        for data_name in test_datanames:
            self.eval_metrics_test[data_name] = DualMetrics(num_test_rounds, num_dual_iter_test)
            self.eval_metrics_test_non_learned[data_name] = DualMetrics(num_test_rounds, num_dual_iter_test)

    @classmethod
    def from_config(cls, cfg, val_datanames, test_datanames, num_test_rounds, num_dual_iter_test, dual_improvement_slope_test, non_learned_updates_test):
        return cls(
            num_train_rounds = cfg.TRAIN.NUM_ROUNDS,
            num_train_rounds_with_grad = cfg.TRAIN.NUM_ROUNDS_WITH_GRAD,
            num_test_rounds = num_test_rounds,
            var_lp_features = cfg.MODEL.VAR_LP_FEATURES,
            con_lp_features = cfg.MODEL.CON_LP_FEATURES,
            edge_lp_features = cfg.MODEL.EDGE_LP_FEATURES,
            num_dual_iter_train = cfg.TRAIN.NUM_DUAL_ITERATIONS,
            num_dual_iter_test = num_dual_iter_test,
            use_layer_norm = cfg.MODEL.USE_LAYER_NORM,
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
            val_fraction = cfg.DATA.VAL_FRACTION,
            val_datanames = val_datanames,
            test_datanames = test_datanames,
            full_coordinate_ascent = cfg.MODEL.FULL_COORDINATE_ASCENT,
            predict_omega = cfg.MODEL.PREDICT_OMEGA,
            skip_connections = cfg.MODEL.SKIP_CONNECTIONS,
            non_learned_updates_test = non_learned_updates_test)

    def configure_optimizers(self):
        if self.hparams.optimizer_name == 'Adam':
            return Adam(self.parameters(), lr=self.hparams.lr)
        else:
            raise ValueError(f'Optimizer {self.hparams.optimizer_name} not exposed.')

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch.edge_index_var_con = batch.edge_index_var_con.to(device)
        batch.batch_index_con = batch.rhs_vector_batch.to(device) # con batch assignment
        batch.batch_index_edge = batch.edge_index_var_con_batch.to(device) # edge batch assignment
        batch.omega = torch.tensor([self.hparams.omega_initial], device = device)
        solvers, solver_state, per_bdd_sol, per_bdd_lb, dist_weights, valid_edge_mask, gt_sol_edge, gt_sol_var, initial_lbs = sol_utils.init_solver_and_get_states(
            batch, device, 'lp_stats', 0, 1.0, batch.omega)
        batch.valid_edge_mask = valid_edge_mask
        batch.gt_sol_edge = gt_sol_edge
        batch.gt_sol_var = gt_sol_var
        batch.initial_lbs = initial_lbs

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
        if not self.hparams.full_coordinate_ascent:
            batch.edge_rest_lp_f = torch.stack((per_bdd_sol, batch.con_coeff.to(device), dist_weights), 1) #TODOAA: Compute moving average of previous dist weights as features?
        else:
            mm_diff = sol_utils.compute_all_min_marginal_diff(solvers, solver_state)
            batch.edge_rest_lp_f = torch.stack((per_bdd_sol, batch.con_coeff.to(device), dist_weights, mm_diff), 1)

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
            metrics_wo_time = {}
            for metric_name_itr_time, val in metric_value.items():
                name_itr, ctime = metric_name_itr_time.split('_time_')
                metrics_wo_time[name_itr] = val
            self.logger.experiment.add_scalars(f'{mode}/{metric_name}', metrics_wo_time, global_step = self.global_step, walltime = float(ctime))
        self.logger.experiment.flush()
        metrics_calculator.reset()
        return metrics_dict['loss']

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

    def training_epoch_end(self, outputs):
        self.log_metrics(self.train_metrics, 'train')

    def validation_epoch_end(self, outputs):
        for data_name in self.val_datanames:
            if self.non_learned_updates_val: # Computes baseline via non-learned updates.
                self.log_metrics(self.eval_metrics_val_non_learned[data_name], f'val_{data_name}_non_learned')
            else:
                losses = self.log_metrics(self.eval_metrics_val[data_name], f'val_{data_name}')
                prev_itr = -1
                best_loss_value = 0
                for name_itr_time, value in losses.items():
                    # self.logger.experiment.add_scalar(f'{mode}/{metric_name}{suffix}', value, global_step = int(itr.replace('itr_', '')))
                    name, itr_time = name_itr_time.split('itr_')
                    itr, time = itr_time.split('_time_')
                    if (prev_itr < int(itr)):
                        prev_itr = int(itr)
                        best_loss_value = value
                self.log('val_loss', best_loss_value)

        self.non_learned_updates_val = False

    def test_epoch_end(self, outputs): # Logs per dataset metrics. Instance level metrics are logged already in test-step(..).
        for data_name in self.test_datanames:
            self.log_metrics_test(self.eval_metrics_test[data_name], f'test_{data_name}', 'learned')
            if self.non_learned_updates_test:
                self.log_metrics_test(self.eval_metrics_test_non_learned[data_name], f'test_{data_name}', 'non_learned')

    def try_concat_gt_edge_solution(self, batch, is_training):
        edge_sol = None
        for file_path, current_sol in zip(batch.file_path, batch.gt_sol_edge):
            if current_sol is None:
                assert current_sol is not None or not is_training, f'gt solution should be known for files: {file_path}'
                return edge_sol
        return torch.cat(batch.gt_sol_edge, 0)

    def dual_loss_mm(self, mm_pred, gt_lp_sol_edge, valid_edge_mask, edge_index_batch):
        if gt_lp_sol_edge is None:
            return None

         # Gather the solution w.r.t all valid edges
        mm_pred_valid = torch.nn.Tanh()(10.0 * mm_pred[valid_edge_mask])
        gt_hi = gt_lp_sol_edge >= 1.0 - 1e-8 # Here mm_diff should ideally be < 0.
        gt_lo = gt_lp_sol_edge <= 0.0 + 1e-8 # Here mm_diff should ideally be > 0.
        gt_frac = torch.logical_and(~gt_hi, ~gt_lo) # Here mm_diff should ideally be 0.
        mm_hi_pred = mm_pred_valid * gt_hi + self.hparams.loss_margin
        mm_lo_pred = -(mm_pred_valid * gt_lo - self.hparams.loss_margin)
        loss_per_edge = torch.relu(mm_hi_pred) + torch.relu(mm_lo_pred) + mm_pred_valid * gt_frac
        loss_per_instance = scatter_mean(loss_per_edge, edge_index_batch[valid_edge_mask]) * len(loss_per_edge)
        return loss_per_instance.sum()

    def dual_loss_lb(self, lb_after_dist, batch_index_con):
        loss_per_instance = scatter_mean(lb_after_dist, batch_index_con) * len(lb_after_dist)
        return -loss_per_instance.sum()

    def single_dual_round(self, batch, num_dual_iterations, improvement_slope, grad_dual_itr_max_itr):
        if not self.hparams.full_coordinate_ascent:
            batch.solver_state, batch.var_lp_f, batch.con_lp_f, batch.edge_rest_lp_f, dist_weights, omega_vec = self.dual_block(
                                                                    batch.solvers, batch.var_lp_f, batch.con_lp_f, 
                                                                    batch.solver_state, batch.edge_rest_lp_f, 
                                                                    batch.var_learned_f, batch.con_learned_f, batch.edge_learned_f, 
                                                                    batch.omega, batch.edge_index_var_con,
                                                                    num_dual_iterations, grad_dual_itr_max_itr, improvement_slope, batch.valid_edge_mask,
                                                                    batch.batch_index_edge)

            batch.solver_state['def_mm'][~batch.valid_edge_mask] = 0 # Locations of terminal nodes can contain nans.
            try:
                assert(torch.all(torch.isfinite(batch.solver_state['def_mm'])))
            except:
                breakpoint()
        else:
            batch.solver_state, batch.var_lp_f, batch.con_lp_f, batch.edge_rest_lp_f = self.dual_block(
                                                                    batch.solvers, batch.var_lp_f, batch.con_lp_f, 
                                                                    batch.solver_state, batch.edge_rest_lp_f, 
                                                                    batch.var_learned_f, batch.con_learned_f, batch.edge_learned_f, 
                                                                    batch.edge_index_var_con, num_dual_iterations)

        return batch, dist_weights, omega_vec

    def dual_rounds(self, batch, num_rounds, num_dual_iterations, improvement_slope, grad_dual_itr_max_itr, is_training = False, instance_log_name = None, non_learned_updates = False):
        if not non_learned_updates:
            batch.var_learned_f, batch.con_learned_f, batch.edge_learned_f = self.lp_feature_extractor(batch.var_lp_f, batch.con_lp_f, batch.solver_state, batch.edge_rest_lp_f, batch.edge_index_var_con)
        loss = 0
        logs = []
        for r in range(num_rounds):
            with torch.set_grad_enabled(r >= num_rounds - self.hparams.num_train_rounds_with_grad and is_training):
                if not non_learned_updates:
                    batch, dist_weights, omega_vec = self.single_dual_round(batch, num_dual_iterations, improvement_slope, grad_dual_itr_max_itr)
                    if instance_log_name is not None:
                        self.log_dist_weights(dist_weights, instance_log_name, batch.edge_index_var_con, (r + 1) * num_dual_iterations)
                        self.log_omega_vector(omega_vec, instance_log_name, batch.edge_index_var_con, (r + 1) * num_dual_iterations)
                else:
                    with torch.no_grad():
                        batch = sol_utils.non_learned_updates(batch, num_dual_iterations, improvement_slope = 0.0, omega = batch.omega.item())

                solver_state = sol_utils.distribute_delta(batch.solvers, batch.solver_state)
                lb_after_dist = sol_utils.compute_per_bdd_lower_bound(batch.solvers, solver_state)
                current_loss = self.dual_loss_lb(lb_after_dist, batch.batch_index_con)
                logs.append({'r' : r, 'lb_per_bdd': lb_after_dist.detach(), 't': time.time()})
                if current_loss is not None:
                    logs[-1]['loss'] = current_loss.detach()
                    loss = loss + torch.pow(torch.tensor(self.hparams.loss_discount_factor), num_rounds - r - 1) * current_loss
        return loss, batch, logs

    def training_step(self, batch, batch_idx):
        num_rounds = self.compute_training_start_round() + 1
        loss, batch, logs = self.dual_rounds(batch, num_rounds, self.hparams.num_dual_iter_train, self.hparams.dual_improvement_slope_train, self.hparams.grad_dual_itr_max_itr, is_training = True)
        self.train_metrics.update(batch, logs)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx = 0):
        data_name = self.val_datanames[dataset_idx]
        if self.non_learned_updates_val:
            loss, batch, logs = self.dual_rounds(batch, self.hparams.num_test_rounds, self.hparams.num_dual_iter_test, self.hparams.dual_improvement_slope_test, 0, is_training = False, non_learned_updates = True)
            self.eval_metrics_val_non_learned[data_name].update(batch, logs)
        else:
            instance_name = os.path.basename(batch.file_path[0])
            data_name = self.val_datanames[dataset_idx]
            instance_log_name = f'val_{data_name}_{self.global_step}/{instance_name}'
            loss, batch, logs = self.dual_rounds(batch, self.hparams.num_test_rounds, self.hparams.num_dual_iter_test, self.hparams.dual_improvement_slope_test, 0, is_training = False, instance_log_name = instance_log_name)
            self.eval_metrics_val[data_name].update(batch, logs)

        return loss

    def test_step(self, batch, batch_idx, dataset_idx = 0):
        assert len(batch.file_path) == 1, 'batch size 1 required for testing.'
        instance_name = os.path.basename(batch.file_path[0])
        data_name = self.test_datanames[dataset_idx]
        instance_log_name = f'test_{data_name}/{instance_name}'
        if self.non_learned_updates_test:
            orig_batch = batch.clone()
        loss, batch, logs = self.dual_rounds(batch, self.hparams.num_test_rounds, self.hparams.num_dual_iter_test, self.hparams.dual_improvement_slope_test, 0, is_training = False, instance_log_name = instance_log_name, non_learned_updates = False)
        instance_level_metrics = DualMetrics(self.hparams.num_test_rounds, self.hparams.num_dual_iter_test).to(batch.edge_index_var_con.device)
        instance_level_metrics.update(batch, logs)
        self.log_metrics_test(instance_level_metrics, f'test_{data_name}_{instance_name}', 'learned')
        self.eval_metrics_test[data_name].update(batch, logs)
        if self.non_learned_updates_test:
            # Perform non-learned updates:
            _, batch, logs = self.dual_rounds(orig_batch, self.hparams.num_test_rounds, self.hparams.num_dual_iter_test, self.hparams.dual_improvement_slope_test, 0, is_training = False, instance_log_name = None, non_learned_updates = True)
            instance_level_metrics = DualMetrics(self.hparams.num_test_rounds, self.hparams.num_dual_iter_test).to(batch.edge_index_var_con.device)
            instance_level_metrics.update(batch, logs)
            self.log_metrics_test(instance_level_metrics, f'test_{data_name}_{instance_name}', 'non_learned')
            self.eval_metrics_test_non_learned[data_name].update(batch, logs)

        return loss