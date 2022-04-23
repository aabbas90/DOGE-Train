import torch
from torch_scatter.scatter import scatter_mean, scatter_sum
from pytorch_lightning.core.lightning import LightningModule
from typing import List, Set, Dict, Tuple, Optional
from torch.optim import Adam
import logging, os, time
import numpy as np
from model.model import FeatureExtractor, DualDistWeightsBlock, DualFullCoordinateAscent
import model.solver_utils as sol_utils
from metrics.dual_metrics import DualMetrics 
from metrics.primal_metrics import PrimalMetrics 

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
                var_lp_features: List[str],
                con_lp_features: List[str],
                edge_lp_features: List[str],
                var_lp_features_init: List[str],
                con_lp_features_init: List[str],
                edge_lp_features_init: List[str],
                num_learned_var_f: int, 
                num_learned_con_f: int, 
                num_learned_edge_f: int,
                num_hidden_layers_edge: int, 
                feature_extractor_depth: int,
                dual_predictor_depth: int,
                optimizer_name: str,
                start_episodic_training_after_epoch: int,
                val_fraction: List[int],
                use_layer_norm: True,
                use_net_solver_costs: False,
                free_update_loss_weight: Optional[float] = 0.0, 
                num_journeys: Optional[int] = 10,
                use_lstm_var: Optional[bool] = False,
                use_rel_gap_loss: Optional[bool] = False,
                predict_dist_weights: Optional[bool] = True,
                predict_omega: Optional[bool] = True,
                free_update: Optional[bool] = False,
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
                'use_rel_gap_loss',
                'use_layer_norm',
                'omega_initial',
                'loss_discount_factor',
                'loss_margin',
                'var_lp_features',
                'con_lp_features',
                'edge_lp_features',
                'var_lp_features_init',
                'con_lp_features_init',
                'edge_lp_features_init',
                'num_learned_var_f', 
                'num_learned_con_f', 
                'num_learned_edge_f',
                'num_hidden_layers_edge',
                'feature_extractor_depth', 
                'dual_predictor_depth', 
                'use_lstm_var',
                'optimizer_name',
                'full_coordinate_ascent',
                'free_update_loss_weight',
                'predict_dist_weights',
                'predict_omega',
                'free_update',
                'use_net_solver_costs',
                'start_episodic_training_after_epoch',
                'num_journeys',
                'val_fraction',
                'val_datanames',
                'test_datanames')

        if not full_coordinate_ascent:
            self.dual_block = DualDistWeightsBlock(
                            var_lp_f_names = var_lp_features,
                            con_lp_f_names = con_lp_features, 
                            edge_lp_f_names = edge_lp_features,
                            depth = dual_predictor_depth,
                            var_dim = num_learned_var_f, 
                            con_dim = num_learned_con_f,
                            edge_dim = num_learned_edge_f,
                            use_layer_norm = use_layer_norm,
                            predict_dist_weights = predict_dist_weights,
                            predict_omega = predict_omega,
                            num_hidden_layers_edge = num_hidden_layers_edge,
                            use_net_solver_costs = use_net_solver_costs,
                            use_lstm_var = use_lstm_var,
                            free_update = free_update,
                            history_num_itr = num_dual_iter_train,
                            free_update_loss_weight = free_update_loss_weight)
        else:
            self.dual_block = DualFullCoordinateAscent(
                            var_lp_f_names = var_lp_features,
                            con_lp_f_names = con_lp_features, 
                            edge_lp_f_names = edge_lp_features,
                            depth = dual_predictor_depth,
                            var_dim = num_learned_var_f, 
                            con_dim = num_learned_con_f,
                            edge_dim = num_learned_edge_f,
                            use_layer_norm = use_layer_norm,
                            num_hidden_layers_edge = num_hidden_layers_edge)

        self.val_datanames = val_datanames
        self.test_datanames = test_datanames
        self.console_logger = logging.getLogger('lightning')
        self.train_log_every_n_epoch = 5
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
        self.logged_hparams = False
        for data_name in test_datanames:
            self.eval_metrics_test[data_name] = DualMetrics(num_test_rounds, num_dual_iter_test)
            self.eval_metrics_test_non_learned[data_name] = DualMetrics(num_test_rounds, num_dual_iter_test)
        self.primal_metrics_learned_dual_test = torch.nn.ModuleDict()
        self.primal_metrics_non_learned_dual_test = torch.nn.ModuleDict()
        for data_name in test_datanames:
            self.primal_metrics_learned_dual_test[data_name] = PrimalMetrics(num_test_rounds, None, on_baseline = True)
            self.primal_metrics_non_learned_dual_test[data_name] = PrimalMetrics(num_test_rounds, None, on_baseline = True)

    @classmethod
    def from_config(cls, cfg, val_datanames, test_datanames, num_test_rounds, num_dual_iter_test, dual_improvement_slope_test, non_learned_updates_test):
        return cls(
            num_train_rounds = cfg.TRAIN.NUM_ROUNDS,
            num_train_rounds_with_grad = cfg.TRAIN.NUM_ROUNDS_WITH_GRAD,
            num_test_rounds = num_test_rounds,
            var_lp_features = cfg.MODEL.VAR_LP_FEATURES,
            con_lp_features = cfg.MODEL.CON_LP_FEATURES,
            edge_lp_features = cfg.MODEL.EDGE_LP_FEATURES,
            var_lp_features_init = cfg.MODEL.VAR_LP_FEATURES_INIT,
            con_lp_features_init = cfg.MODEL.CON_LP_FEATURES_INIT,
            edge_lp_features_init = cfg.MODEL.EDGE_LP_FEATURES_INIT,
            num_hidden_layers_edge = cfg.MODEL.NUM_HIDDEN_LAYERS_EDGE,
            use_lstm_var = cfg.MODEL.USE_LSTM_VAR,
            num_dual_iter_train = cfg.TRAIN.NUM_DUAL_ITERATIONS,
            num_dual_iter_test = num_dual_iter_test,
            use_layer_norm = cfg.MODEL.USE_LAYER_NORM,
            use_net_solver_costs = cfg.MODEL.USE_NET_SOLVER_COSTS,
            use_rel_gap_loss = cfg.TRAIN.USE_RELATIVE_GAP_LOSS,
            dual_improvement_slope_train = cfg.TRAIN.DUAL_IMPROVEMENT_SLOPE,
            dual_improvement_slope_test = dual_improvement_slope_test,
            grad_dual_itr_max_itr = cfg.TRAIN.GRAD_DUAL_ITR_MAX_ITR,
            lr = cfg.TRAIN.BASE_LR,
            loss_discount_factor = cfg.TRAIN.LOSS_DISCOUNT_FACTOR,
            loss_margin = cfg.TRAIN.LOSS_MARGIN,
            omega_initial = cfg.MODEL.OMEGA_INITIAL,
            free_update = cfg.MODEL.FREE_UPDATE,
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
            predict_dist_weights = cfg.MODEL.PREDICT_DIST_WEIGHTS,
            predict_omega = cfg.MODEL.PREDICT_OMEGA,
            non_learned_updates_test = non_learned_updates_test,
            num_journeys = cfg.TRAIN.NUM_JOURNEYS)

    def configure_optimizers(self):
        if self.hparams.optimizer_name == 'Adam':
            return Adam(self.parameters(), lr=self.hparams.lr)
        else:
            raise ValueError(f'Optimizer {self.hparams.optimizer_name} not exposed.')

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        gt_type = None # 'lp_stats' TODOAA
        batch, dist_weights = sol_utils.init_solver_and_get_states(batch, device, gt_type,
                    self.hparams.var_lp_features_init, self.hparams.con_lp_features_init, self.hparams.edge_lp_features_init, 
                    self.hparams.num_dual_iter_train * 2, 0.0, self.hparams.omega_initial, 
                    distribute_deltaa = self.hparams.full_coordinate_ascent, num_grad_iterations_dual_features = 0, 
                    compute_history_for_itrs = self.hparams.num_dual_iter_train)
        batch.dist_weights = dist_weights
        batch.objective_dev = batch.objective.to(device).to(torch.float64)
        batch.initial_lb_per_instance = scatter_sum(batch.con_lp_f[:, self.hparams.con_lp_features.index('lb')], batch.batch_index_con)
        if self.hparams.use_lstm_var:
            batch.var_hidden_states_lstm = {'h': torch.zeros((batch.var_lp_f.shape[0], self.hparams.num_learned_var_f), device = device),
                                            'c': torch.zeros((batch.var_lp_f.shape[0], self.hparams.num_learned_var_f), device = device)}
        else:
            batch.var_hidden_states_lstm = torch.empty((0, 0))

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

    def compute_training_start_round(self):
        current_start_epoch = self.current_epoch - self.hparams.start_episodic_training_after_epoch
        max_training_epoch_mod = 1 + (self.trainer.max_epochs // self.hparams.num_journeys) # Divides num_journeys many journeys.
        fraction = float(current_start_epoch) / max_training_epoch_mod
        if fraction < 0:
            return max(0, self.hparams.num_train_rounds_with_grad - 1)
        fraction = fraction % 1
        fraction = fraction * fraction
        mean_start_step = fraction * (self.hparams.num_train_rounds)
        proposed_start_step = np.round(np.random.normal(mean_start_step, 3)).astype(np.int32).item(0)
        proposed_start_step = max(max(min(proposed_start_step, self.hparams.num_train_rounds - 1), 0), self.hparams.num_train_rounds_with_grad - 1)
        self.logger.experiment.add_scalar('train/start_grad_round', proposed_start_step, global_step = self.global_step)
        return proposed_start_step

    def log_metrics(self, metrics_calculator, mode, log_to_tb = True):
        metrics_dict, _, _ = metrics_calculator.compute()
        best_pred_lb = np.NINF
        last_itr = 0
        for metric_name, metric_value in metrics_dict.items():
            metrics_wo_time = {}
            for metric_name_itr_time, val in metric_value.items():
                name_itr, ctime = metric_name_itr_time.split('_time_')
                _, itr = name_itr.split('itr_')
                metrics_wo_time[name_itr] = val
                if 'lower_bounds' in metric_name and 'pred_' in name_itr and not 'pred_clip_' in name_itr: #and int(itr) == self.hparams.num_train_rounds * self.hparams.num_dual_iter_train:
                    best_pred_lb = max(val.item(), best_pred_lb)
            if log_to_tb:
                self.logger.experiment.add_scalars(f'{mode}/{metric_name}', metrics_wo_time, global_step = self.global_step, walltime = float(ctime))
        self.logger.experiment.flush()
        if np.isfinite(best_pred_lb):
            self.log('train_last_round_lb', best_pred_lb)
            if not self.logged_hparams:
                self.logger.log_hyperparams(self.hparams, {"train_last_round_lb": best_pred_lb})
                self.logged_hparams = True
        metrics_calculator.reset()
        return metrics_dict['loss']

    def log_metrics_test(self, metrics_calculator, prefix, suffix):
        assert('test' in prefix)
        metrics_dict, max_lb_per_instance, gt_time_mean = metrics_calculator.compute()
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

        if gt_time_mean is not None:
            self.logger.experiment.add_scalar(f'{prefix}/gt_lb_compute_time', gt_time_mean, global_step = 0)
            print(f'\t ground truth LB compute time: {gt_time_mean}')
        self.logger.experiment.flush()
        metrics_calculator.reset()
        return max_lb_per_instance

    def log_primal_metrics_test(self, metrics_calculator, prefix, suffix):
        assert('test' in prefix)
        metrics_dict, min_cost_per_instance, gt_time_mean = metrics_calculator.compute()
        print(f'\n{prefix}')
        for metric_name, metric_value_per_round_dict in metrics_dict.items():
            if 'num_disagreements' in metric_name:
                continue
            print(f'\t {metric_name}_{suffix}: ')
            prev_value = None
            for round, value in metric_value_per_round_dict.items():
                self.logger.experiment.add_scalar(f'{prefix}/{metric_name}_{suffix}', value, global_step = int(round.replace('round_', '')))
                if prev_value is None or prev_value != value:
                    print(f'\t \t {round}: {value}')
                prev_value = value
        if gt_time_mean is not None:
            self.logger.experiment.add_scalar(f'{prefix}/gt_obj_compute_time', gt_time_mean, global_step = 0)
            print(f'\t ground truth OBJ compute time: {gt_time_mean}')
        self.logger.experiment.flush()
        metrics_calculator.reset()
        return min_cost_per_instance

    def log_primal_dual_gap(self, max_lb_per_inst_dict, min_cost_per_inst_dict, prefix, suffix):
        pd_gap_sum = 0.0
        num = 0
        print(f'\n{prefix}')
        for filename, lb in max_lb_per_inst_dict.items():
            assert filename in min_cost_per_inst_dict
            cost = min_cost_per_inst_dict[filename]
            filename_wo_ext = os.path.splitext(filename)[0]
            if np.isfinite(pd_gap_sum):
                if cost * lb < 0:
                    gap = np.inf
                    pd_gap_sum = np.inf
                else:
                    gap = max((cost - lb) / (np.abs(lb) + 1e-6), 0.0)
                    print(f'\t \t pd_gap_{filename_wo_ext}_{suffix}: {gap}')
                    self.logger.experiment.add_scalar(f'{prefix}/pd_gap_{filename_wo_ext}_{suffix}', gap, global_step = 0)
                    pd_gap_sum += gap
            num += 1
        pd_gap_mean = pd_gap_sum / num
        print(f'\t mean_pd_gap_{suffix}: {pd_gap_mean}')
        self.logger.experiment.add_scalar(f'{prefix}/pd_gap_{suffix}', pd_gap_mean, global_step = 0)

    def log_grad_norm(self, grad_norm_dict: Dict[str, float]) -> None:
        gn = grad_norm_dict['grad_2.0_norm_total']
        # if gn > 100:
        #     print(f'Itr: {self.global_step}, grad norm: {gn}')
        self.logger.experiment.add_scalar('train/grad_norm', grad_norm_dict['grad_2.0_norm_total'], global_step = self.global_step)
        self.logger.experiment.add_scalar('train/progress_percent.', 100 * self.current_epoch / self.trainer.max_epochs, global_step = self.global_step)

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

    def on_train_epoch_start(self):
        self.num_rounds = self.compute_training_start_round() + 1

    def training_epoch_end(self, outputs):
        self.log_metrics(self.train_metrics, 'train', self.current_epoch % self.train_log_every_n_epoch == 0)

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
            max_lb_per_inst_learned = self.log_metrics_test(self.eval_metrics_test[data_name], f'test_{data_name}', 'learned')
            if self.non_learned_updates_test:
                max_lb_per_inst_non_learned = self.log_metrics_test(self.eval_metrics_test_non_learned[data_name], f'test_{data_name}', 'non_learned')
                min_cost_per_inst_non_learned = self.log_primal_metrics_test(self.primal_metrics_non_learned_dual_test[data_name], f'test_primal_{data_name}', 'non_learned_dual')
                self.log_primal_dual_gap(max_lb_per_inst_non_learned, min_cost_per_inst_non_learned, f'test_primal_dual_{data_name}', 'non_learned_dual')
            min_cost_per_inst_learned = self.log_primal_metrics_test(self.primal_metrics_learned_dual_test[data_name], f'test_primal_{data_name}', 'learned_dual')
            self.log_primal_dual_gap(max_lb_per_inst_learned, min_cost_per_inst_learned, f'test_primal_dual_{data_name}', 'learned_dual')

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

    def single_dual_round(self, batch, num_dual_iterations, improvement_slope, grad_dual_itr_max_itr, return_best_dual = False, current_max_lb = None, best_solver_state = None):
        lb_after_free_update = None
        if not self.hparams.full_coordinate_ascent:
            new_solver_state, batch.var_lp_f, batch.con_lp_f, batch.edge_rest_lp_f, dist_weights, omega_vec, batch.var_hidden_states_lstm, lb_after_free_update = self.dual_block(
                                                                    batch.solvers, batch.var_lp_f, batch.con_lp_f, 
                                                                    batch.solver_state, batch.edge_rest_lp_f, 
                                                                    batch.omega, batch.edge_index_var_con,
                                                                    num_dual_iterations, grad_dual_itr_max_itr, improvement_slope, batch.valid_edge_mask,
                                                                    batch.batch_index_var, batch.batch_index_con, batch.batch_index_edge, 
                                                                    self.logger.experiment, batch.file_path, self.global_step, batch.objective_dev,
                                                                    batch.var_hidden_states_lstm, batch.dist_weights)

            new_solver_state['def_mm'][~batch.valid_edge_mask] = 0 # Locations of terminal nodes can contain nans.
            if return_best_dual:
                new_lb = batch.solvers[0].lower_bound()
                assert len(batch.solvers) == 1
                if current_max_lb is None or new_lb > current_max_lb:
                    return batch, dist_weights, omega_vec, new_solver_state, new_lb, lb_after_free_update
            batch.solver_state = new_solver_state
        else:
            batch.solver_state, batch.var_lp_f, batch.con_lp_f, batch.edge_rest_lp_f, dist_weights, omega_vec = self.dual_block(
                                                                                    batch.solvers, batch.var_lp_f, batch.con_lp_f, 
                                                                                    batch.solver_state, batch.edge_rest_lp_f, 
                                                                                    batch.var_learned_f, batch.con_learned_f, batch.edge_learned_f, 
                                                                                    batch.edge_index_var_con, num_dual_iterations)

        return batch, dist_weights, omega_vec, best_solver_state, current_max_lb, lb_after_free_update

    def dual_rounds(self, batch, num_rounds, num_dual_iterations, improvement_slope, grad_dual_itr_max_itr, is_training = False, instance_log_name = None, non_learned_updates = False, return_best_dual = False):
        loss = 0
        logs = [{'r' : 0, 'lb_per_instance': scatter_sum(batch.con_lp_f[:, self.hparams.con_lp_features.index('lb')], batch.batch_index_con), 't': time.time()}]
        gt_obj_normalized = None
        if 'gt_obj_normalized' in batch.keys:
            gt_obj_normalized = batch.gt_obj_normalized
        current_loss = self.dual_loss_lb(batch.con_lp_f[:, self.hparams.con_lp_features.index('lb')], batch.batch_index_con, batch.initial_lb_per_instance, gt_obj_normalized)
        if current_loss is not None:
            logs[-1]['loss'] = current_loss.detach()
        non_grad_lb_per_instance = logs[0]['lb_per_instance']
        if not is_training and instance_log_name is not None:
            device = batch.solver_state['hi_costs'].device
            max_diff, mean_diff = sol_utils.dual_feasbility_check(batch.solvers, batch.solver_state, batch.objective.to(device), batch.num_vars, True)
            self.logger.experiment.add_scalar(f'{instance_log_name}/feasibility_check', max_diff, global_step = 0)
            self.logger.experiment.add_scalar(f'{instance_log_name}/feasibility_check_mean', mean_diff, global_step = 0)
        current_max_lb = np.NINF
        best_solver_state = None
        lb_after_free_update = None
        for r in range(num_rounds):
            if 'round_index' in self.hparams.con_lp_features:
                batch.con_lp_f[:, self.hparams.con_lp_features.index('round_index')] = r
            grad_enabled = r >= num_rounds - self.hparams.num_train_rounds_with_grad and is_training
            with torch.set_grad_enabled(grad_enabled):
                if not non_learned_updates:
                    batch, dist_weights, omega_vec, best_solver_state, current_max_lb, lb_after_free_update = self.single_dual_round(batch, num_dual_iterations, 0.0, grad_dual_itr_max_itr, 
                                                                                                            return_best_dual, current_max_lb, best_solver_state)
                    if instance_log_name is not None:
                        self.log_dist_weights(dist_weights, instance_log_name, batch.edge_index_var_con, (r + 1) * num_dual_iterations)
                        self.log_omega_vector(omega_vec, instance_log_name, batch.edge_index_var_con, (r + 1) * num_dual_iterations)
                else:
                    with torch.no_grad():
                        batch = sol_utils.non_learned_updates(batch, self.hparams.edge_lp_features, num_dual_iterations, improvement_slope = 0.0, omega = batch.omega.item())

                if self.hparams.free_update_loss_weight < 1.0:
                    solver_state = sol_utils.distribute_delta(batch.solvers, batch.solver_state)
                    lb_after_dist = sol_utils.compute_per_bdd_lower_bound(batch.solvers, solver_state)
                    current_loss = (1.0 - self.hparams.free_update_loss_weight) * self.dual_loss_lb(lb_after_dist, batch.batch_index_con, batch.initial_lb_per_instance, gt_obj_normalized)
                else:
                    current_loss = 0
                if lb_after_free_update is not None:
                    current_loss = current_loss + self.hparams.free_update_loss_weight * self.dual_loss_lb(lb_after_free_update, batch.batch_index_con, batch.initial_lb_per_instance, gt_obj_normalized)
                with torch.no_grad():
                    logs.append({'r' : r + 1, 'lb_per_instance': scatter_sum(lb_after_dist, batch.batch_index_con), 't': time.time()})
                if current_loss is not None:
                    logs[-1]['loss'] = current_loss.detach()
                    loss = loss + torch.pow(torch.tensor(self.hparams.loss_discount_factor), num_rounds - r - 1) * current_loss
            if not is_training and instance_log_name is not None:
                device = batch.solver_state['hi_costs'].device
                max_diff, mean_diff = sol_utils.dual_feasbility_check(batch.solvers, solver_state, batch.objective.to(device), batch.num_vars)
                self.logger.experiment.add_scalar(f'{instance_log_name}/feasibility_check', max_diff, global_step = (r + 1) * num_dual_iterations)
                self.logger.experiment.add_scalar(f'{instance_log_name}/feasibility_check_mean', mean_diff, global_step = (r + 1) * num_dual_iterations)
            if not grad_enabled:
                non_grad_lb_per_instance = logs[-1]['lb_per_instance']

        if is_training:
            self.logger.experiment.add_scalar('train/dist_weights_edge_mlp_w_mean', self.dual_block.dist_weights_predictor.edge_mlp[2].weight[0].mean(), global_step = self.global_step)
            self.logger.experiment.add_scalar('train/dist_weights_edge_mlp_w_std', self.dual_block.dist_weights_predictor.edge_mlp[2].weight[0].std(), global_step = self.global_step)
            self.logger.experiment.add_scalar('train/subgradient_step_size', torch.abs(self.dual_block.subgradient_step_size[0]).item(), global_step = self.global_step)
            min_lb_change_per_instance = torch.min(logs[-1]['lb_per_instance'] - non_grad_lb_per_instance)
            self.logger.experiment.add_scalar('train/min_lb_change_per_instance', min_lb_change_per_instance, global_step = self.global_step)
        return loss, batch, logs, best_solver_state

    def training_step(self, batch, batch_idx):
        loss, batch, logs, _ = self.dual_rounds(batch, self.num_rounds, self.hparams.num_dual_iter_train, self.hparams.dual_improvement_slope_train, self.hparams.grad_dual_itr_max_itr, is_training = True)
        self.train_metrics.update(batch, logs)
        #print(f'Itr: {self.global_step}, round: {self.num_rounds}, loss: {loss.item()}, file: {batch.file_path[0]}')
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx = 0):
        data_name = self.val_datanames[dataset_idx]
        if self.non_learned_updates_val:
            loss, batch, logs, _ = self.dual_rounds(batch, self.hparams.num_test_rounds, self.hparams.num_dual_iter_test, self.hparams.dual_improvement_slope_test, 0, is_training = False, non_learned_updates = True)
            self.eval_metrics_val_non_learned[data_name].update(batch, logs)
        else:
            instance_name = os.path.basename(batch.file_path[0])
            data_name = self.val_datanames[dataset_idx]
            instance_log_name = f'val_{data_name}_{self.global_step}/{instance_name}'
            loss, batch, logs, _ = self.dual_rounds(batch, self.hparams.num_test_rounds, self.hparams.num_dual_iter_test, self.hparams.dual_improvement_slope_test, 0, is_training = False, instance_log_name = instance_log_name)
            self.eval_metrics_val[data_name].update(batch, logs)

        return loss

    def test_step(self, batch, batch_idx, dataset_idx = 0):
        assert len(batch.file_path) == 1, 'batch size 1 required for testing.'
        instance_name = os.path.basename(batch.file_path[0])
        data_name = self.test_datanames[dataset_idx]
        instance_log_name = f'test_{data_name}_{instance_name}'
        if self.non_learned_updates_test:
            orig_batch = batch.clone()
        loss, batch, logs, best_solver_state = self.dual_rounds(batch, self.hparams.num_test_rounds, self.hparams.num_dual_iter_test, self.hparams.dual_improvement_slope_test, 0, 
                                            is_training = False, instance_log_name = instance_log_name, non_learned_updates = False, return_best_dual = True)
        instance_level_metrics = DualMetrics(self.hparams.num_test_rounds, self.hparams.num_dual_iter_test).to(batch.edge_index_var_con.device)
        instance_level_metrics.update(batch, logs)
        self.log_metrics_test(instance_level_metrics, f'test_{data_name}_{instance_name}', 'learned')
        self.eval_metrics_test[data_name].update(batch, logs)

        batch.solver_state = best_solver_state
        batch.solver_state = sol_utils.distribute_delta(batch.solvers, batch.solver_state)
        self.test_primal(batch, True, data_name, instance_name)
        best_solver_state_non_learned = None
        if self.non_learned_updates_test:
            # Perform non-learned updates:
            _, batch, logs, _ = self.dual_rounds(orig_batch, self.hparams.num_test_rounds, self.hparams.num_dual_iter_test, self.hparams.dual_improvement_slope_test, 0, 
                                                is_training = False, instance_log_name = None, non_learned_updates = True)
            instance_level_metrics = DualMetrics(self.hparams.num_test_rounds, self.hparams.num_dual_iter_test).to(batch.edge_index_var_con.device)
            instance_level_metrics.update(batch, logs)
            self.log_metrics_test(instance_level_metrics, f'test_{data_name}_{instance_name}', 'non_learned')
            self.eval_metrics_test_non_learned[data_name].update(batch, logs)

            batch.solver_state = sol_utils.distribute_delta(batch.solvers, batch.solver_state)
            self.test_primal(batch, False, data_name, instance_name)
        return loss

    def test_primal(self, batch, learned_dual, data_name, instance_name):
        num_rounds = 200
        mm_diff, sol, logs = sol_utils.primal_rounding_non_learned(num_rounds, batch.solvers, batch.solver_state, batch.obj_multiplier, batch.obj_offset, 500, 1e-6, 0.5, batch.edge_index_var_con, batch.dist_weights, init_delta = 1.0, delta_growth_rate = 1.2)
        instance_level_metrics = PrimalMetrics(num_rounds, None, on_baseline=True).to(batch.edge_index_var_con.device)
        instance_level_metrics.update(batch, logs)
        if learned_dual:
            self.log_primal_metrics_test(instance_level_metrics, f'test_{data_name}_{instance_name}', 'learned_dual')
            self.primal_metrics_learned_dual_test[data_name].update(batch, logs)
        else:
            self.log_primal_metrics_test(instance_level_metrics, f'test_{data_name}_{instance_name}', 'non_learned_dual')
            self.primal_metrics_non_learned_dual_test[data_name].update(batch, logs)

