import torch
from torch_scatter.scatter import scatter_sum, scatter_mean
from pytorch_lightning.core.lightning import LightningModule
from typing import List, Set, Dict, Tuple, Optional
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import logging, os
import numpy as np
from model.model import FeatureExtractor, PrimalPerturbationBlock
import model.solver_utils as sol_utils
from metrics.primal_metrics import PrimalMetrics 

class PrimalRoundingBDD(LightningModule):
    def __init__(self, 
                num_train_rounds: int,
                num_test_rounds: int,
                num_dual_iter_train: int,
                num_dual_iter_test: int,
                dual_improvement_slope_train: float,
                dual_improvement_slope_test: float,
                grad_dual_itr_max_itr: int,
                lr: float,
                lr_gamma: float,
                loss_discount_factor:float,
                loss_margin: float,
                min_perturbation: float,
                omega: float,
                var_lp_features: List[str],
                con_lp_features: List[str],
                edge_lp_features: List[str],               
                var_lp_features_init: List[str],
                con_lp_features_init: List[str],
                edge_lp_features_init: List[str],
                num_learned_var_f: int, 
                num_learned_con_f: int, 
                num_learned_edge_f: int,
                feature_extractor_depth: int,
                primal_predictor_depth: int,
                optimizer_name: str,
                datasets: List[str],
                val_fraction: List[int],
                start_episodic_training_after_epoch: int,
                num_train_rounds_with_grad: int = 1,
                use_layer_norm: bool = False,
                use_lstm_var: bool = False,
                log_every_n_steps: int = 20,
                val_datanames: Optional[List[str]] = None,
                test_datanames: Optional[List[str]] = None,
                non_learned_updates_test = False
                ):
        super(PrimalRoundingBDD, self).__init__()
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
                'lr_gamma',
                'omega',
                'loss_discount_factor',
                'loss_margin',
                'min_perturbation',
                'var_lp_features',
                'con_lp_features',
                'edge_lp_features',
                'var_lp_features_init',
                'con_lp_features_init',
                'edge_lp_features_init',
                'num_learned_var_f', 
                'num_learned_con_f', 
                'num_learned_edge_f',
                'feature_extractor_depth', 
                'primal_predictor_depth', 
                'use_lstm_var',
                'use_layer_norm',
                'optimizer_name',
                'start_episodic_training_after_epoch',
                'datasets',
                'val_fraction',
                'val_datanames',
                'test_datanames')

        self.lp_feature_extractor = FeatureExtractor(
                        num_var_lp_f = len(var_lp_features), out_var_dim = num_learned_var_f, 
                        num_con_lp_f = len(con_lp_features), out_con_dim = num_learned_con_f,
                        num_edge_lp_f = len(edge_lp_features), out_edge_dim = num_learned_edge_f,
                        depth = feature_extractor_depth, use_layer_norm = use_layer_norm, use_def_mm = False)

        self.primal_block = PrimalPerturbationBlock(
                        var_lp_f_names = var_lp_features,
                        con_lp_f_names = con_lp_features, 
                        edge_lp_f_names = edge_lp_features,
                        depth = primal_predictor_depth,
                        var_dim = num_learned_var_f, 
                        con_dim = num_learned_con_f,
                        edge_dim = num_learned_edge_f,
                        use_layer_norm = use_layer_norm,
                        min_perturbation = min_perturbation,
                        use_lstm_var = use_lstm_var)

        self.val_datanames = val_datanames
        self.test_datanames = test_datanames
        self.non_learned_updates_test = False
        self.log_every_n_steps = log_every_n_steps
        self.non_learned_updates_test = non_learned_updates_test
        self.train_metrics = PrimalMetrics(num_train_rounds, self.hparams.con_lp_features)

        self.eval_metrics_val = torch.nn.ModuleDict()
        self.eval_metrics_val_non_learned = torch.nn.ModuleDict()
        self.non_learned_updates_val = True
        for data_name in val_datanames:
            self.eval_metrics_val[data_name] =PrimalMetrics(num_test_rounds, self.hparams.con_lp_features)
            self.eval_metrics_val_non_learned[data_name] = PrimalMetrics(num_test_rounds, self.hparams.con_lp_features, on_baseline = True)

        self.non_learned_updates_test = non_learned_updates_test
        self.eval_metrics_test = torch.nn.ModuleDict()
        self.eval_metrics_test_non_learned = torch.nn.ModuleDict()
        for data_name in test_datanames:
            self.eval_metrics_test[data_name] = PrimalMetrics(num_test_rounds, self.hparams.con_lp_features)
            self.eval_metrics_test_non_learned[data_name] = PrimalMetrics(num_test_rounds, self.hparams.con_lp_features, on_baseline = True)

    @classmethod
    def from_config(cls, cfg, val_datanames, test_datanames, num_dual_iter_test, num_test_rounds, dual_improvement_slope_test, non_learned_updates_test):
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
            num_dual_iter_train = cfg.TRAIN.NUM_DUAL_ITERATIONS,
            num_dual_iter_test = num_dual_iter_test,
            dual_improvement_slope_train = cfg.TRAIN.DUAL_IMPROVEMENT_SLOPE,
            dual_improvement_slope_test = dual_improvement_slope_test,
            grad_dual_itr_max_itr = cfg.TRAIN.GRAD_DUAL_ITR_MAX_ITR,
            lr = cfg.TRAIN.BASE_LR,
            lr_gamma = cfg.TRAIN.LR_GAMMA,
            loss_discount_factor = cfg.TRAIN.LOSS_DISCOUNT_FACTOR,
            loss_margin = cfg.TRAIN.LOSS_MARGIN,
            min_perturbation = cfg.TRAIN.MIN_PERTURBATION,
            omega = cfg.MODEL.OMEGA,
            num_learned_var_f = cfg.MODEL.VAR_FEATURE_DIM, 
            num_learned_con_f = cfg.MODEL.CON_FEATURE_DIM,
            num_learned_edge_f = cfg.MODEL.EDGE_FEATURE_DIM,
            feature_extractor_depth = cfg.MODEL.FEATURE_EXTRACTOR_DEPTH,
            primal_predictor_depth = cfg.MODEL.PRIMAL_PRED_DEPTH,
            start_episodic_training_after_epoch = cfg.TRAIN.START_EPISODIC_TRAINING_AFTER_EPOCH,
            optimizer_name = cfg.TRAIN.OPTIMIZER,
            use_layer_norm = cfg.MODEL.USE_LAYER_NORM,
            use_lstm_var = cfg.MODEL.USE_LSTM_VAR,
            datasets = cfg.DATA.DATASETS,
            val_fraction = cfg.DATA.VAL_FRACTION,
            log_every_n_steps = cfg.LOG_EVERY,
            val_datanames = val_datanames,
            test_datanames = test_datanames,
            non_learned_updates_test = non_learned_updates_test)

    def configure_optimizers(self):
        if self.hparams.optimizer_name == 'Adam':
            optimizer = Adam(self.parameters(), lr=self.hparams.lr)
            scheduler = MultiStepLR(optimizer, milestones = [15], gamma=self.hparams.lr_gamma)
            return [optimizer], [scheduler]
        else:
            raise ValueError(f'Optimizer {self.hparams.optimizer_name} not exposed.')

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
        if self.hparams.use_lstm_var:
            batch.var_hidden_states_lstm = {'h': torch.zeros((batch.var_lp_f.shape[0], self.hparams.num_learned_var_f), device = device),
                                            'c': torch.zeros((batch.var_lp_f.shape[0], self.hparams.num_learned_var_f), device = device)}
        else:
            batch.var_hidden_states_lstm = torch.empty((0, 0))
        return batch

    def compute_training_start_round(self):
        current_start_epoch = self.current_epoch - self.hparams.start_episodic_training_after_epoch
        max_training_epoch_mod = self.trainer.max_epochs // 10 # Divides into ten journeys.
        fraction = float(current_start_epoch) / max_training_epoch_mod
        if fraction < 0:
            return 0
        fraction = fraction % 1
        fraction = fraction * fraction
        mean_start_step = fraction * (self.hparams.num_train_rounds)
        proposed_start_step = np.round(np.random.normal(mean_start_step, 3)).astype(np.int32).item(0)
        self.logger.experiment.add_scalar('train/start_grad_round', proposed_start_step, global_step = self.global_step)
        return max(min(proposed_start_step, self.hparams.num_train_rounds - 1), 0)

    def log_metrics(self, metrics_calculator, mode):
        metrics_dict = metrics_calculator.compute()
        for metric_name, metric_value_per_round_dict in metrics_dict.items():
            self.logger.experiment.add_scalars(f'{mode}/{metric_name}', metric_value_per_round_dict, global_step = self.global_step)
        self.logger.experiment.flush()
        metrics_calculator.reset()

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

    def training_epoch_end(self, outputs):
        self.log_metrics(self.train_metrics, 'train')
        # sch = self.lr_schedulers()
        # sch.step()

    def validation_epoch_end(self, outputs):
        for data_name in self.val_datanames:
            self.log_metrics(self.eval_metrics_val[data_name], f'val_{data_name}_learned')
            if self.non_learned_updates_val: # Computes baseline via non-learned updates.
                self.log_metrics(self.eval_metrics_val_non_learned[data_name], f'val_{data_name}_non_learned')

        self.non_learned_updates_val = False

    def test_epoch_end(self, outputs):
        for data_name in self.test_datanames:
            self.log_metrics_test(self.eval_metrics_test[data_name], f'test_{data_name}', 'learned')
            if self.non_learned_updates_test:
                self.log_metrics_test(self.eval_metrics_test_non_learned[data_name], f'test_{data_name}', 'non_learned')

    def loss_on_lb_increase(self, con_lp_f, batch_index_con, orig_var_cost_mean, orig_var_cost_std):
        # lb_per_instance = scatter_mean(con_lp_f[:, self.con_lp_f_names.index('prev_lb')], batch_index_con) * len(batch_index_con)
        # return lb_per_instance.sum()
        # Larger problems should have more lb increase so taking sum directly:
        prev_lb_per_instance = scatter_sum(con_lp_f[:, self.hparams.con_lp_features.index('prev_lb')], batch_index_con)
        orig_lb_per_instance = scatter_sum(con_lp_f[:, self.hparams.con_lp_features.index('orig_lb')], batch_index_con) * orig_var_cost_std + orig_var_cost_mean
        return (prev_lb_per_instance - orig_lb_per_instance).mean()  
 
    # def primal_loss(self, mm_pred, gt_ilp_sol_edge, valid_edge_mask, batch_index_edge):
    #     if gt_ilp_sol_edge is None:
    #         return None
    #      # Gather the solution w.r.t all valid edges
    #     mm_pred_valid = mm_pred[valid_edge_mask]

    #     # if gt_ilp_solution > 0 then mm_pred should be < -eps and if gt_ilp_solution == 0 then mm_pred should be > eps:
    #     loss_per_edge = (torch.relu(gt_ilp_sol_edge * (mm_pred_valid + self.hparams.loss_margin)) + torch.relu((gt_ilp_sol_edge - 1.0) * (mm_pred_valid - self.hparams.loss_margin)))
    #     loss_per_instance = scatter_mean(loss_per_edge, batch_index_edge[valid_edge_mask]) * len(loss_per_edge)
    #     return loss_per_instance.sum()

    # def try_concat_gt_edge_solution(self, batch, is_training):
    #     edge_sol = None
    #     for file_path, current_sol in zip(batch.file_path, batch.gt_sol_edge):
    #         if current_sol is None:
    #             assert current_sol is not None or not is_training, f'gt solution should be known for files: {file_path}'
    #             return edge_sol
    #     return torch.cat(batch.gt_sol_edge, 0)

    def single_primal_round(self, batch, num_dual_iterations, improvement_slope, grad_dual_itr_max_itr):
        batch.var_lp_f, batch.con_lp_f, batch.edge_rest_lp_f, batch.var_hidden_states_lstm = self.primal_block(
                                                                batch.solvers, batch.var_lp_f, batch.con_lp_f, batch.solver_state,
                                                                batch.edge_rest_lp_f, 
                                                                batch.var_learned_f, batch.con_learned_f, batch.edge_learned_f, 
                                                                batch.dist_weights, batch.omega, batch.edge_index_var_con,
                                                                num_dual_iterations, grad_dual_itr_max_itr, improvement_slope,
                                                                batch.batch_index_var, batch.batch_index_con, batch.batch_index_edge,
                                                                batch.num_cons, batch.var_hidden_states_lstm)
        return batch

    def primal_rounds(self, batch, num_rounds, num_dual_iterations, improvement_slope, grad_dual_itr_max_itr, is_training = False):
        batch.var_learned_f, batch.con_learned_f, batch.edge_learned_f = self.lp_feature_extractor(batch.var_lp_f, batch.con_lp_f, batch.solver_state, batch.edge_rest_lp_f, batch.edge_index_var_con)
        loss = 0
        logs = []
        # gt_ilp_sol_edge = self.try_concat_gt_edge_solution(batch, is_training)
        for r in range(num_rounds):
            with torch.set_grad_enabled(r >= num_rounds - self.hparams.num_train_rounds_with_grad and is_training):
                batch.con_lp_f[:, self.hparams.con_lp_features.index('round_index')] = r
                batch = self.single_primal_round(batch, num_dual_iterations, improvement_slope, grad_dual_itr_max_itr)
                all_mm_diff = batch.edge_rest_lp_f[:, self.hparams.edge_lp_features.index('prev_mm_diff')].clone().detach()
                pert_lb = batch.con_lp_f[:, self.hparams.con_lp_features.index('prev_lb')].clone().detach()
                current_loss = self.loss_on_lb_increase(batch.con_lp_f, batch.batch_index_con, batch.var_cost_mean, batch.var_cost_std)
                logs.append({'r' : r, 'all_mm_diff': all_mm_diff, 'prev_lb': pert_lb})
                if current_loss is not None:
                    loss = loss + torch.pow(torch.tensor(self.hparams.loss_discount_factor), num_rounds - r - 1) * current_loss
                    logs[-1]['loss'] = current_loss.detach()

        return loss, batch, logs

    def primal_rounds_non_learned(self, batch, num_rounds, num_dual_iterations, improvement_slope):
        assert(len(batch.solvers) == 1)
        all_mm_diff, _, logs = sol_utils.primal_rounding_non_learned(num_rounds, batch.solvers, batch.solver_state, batch.obj_multiplier, batch.obj_offset, num_dual_iterations, improvement_slope, batch.omega.item(), batch.edge_index_var_con, batch.dist_weights)
        return 0, batch, logs

    def training_step(self, batch, batch_idx):
        num_rounds = self.compute_training_start_round() + 1
        loss, batch, logs = self.primal_rounds(batch, num_rounds, self.hparams.num_dual_iter_train,  self.hparams.dual_improvement_slope_train, self.hparams.grad_dual_itr_max_itr, is_training = True)
        losses_dict = {}
        for log_r in logs:
            r = log_r['r']
            losses_dict[f'loss_round_{r}'] = log_r['loss']
        self.logger.experiment.add_scalars(f'train/loss_every_step', losses_dict, global_step = self.global_step)
        if self.global_step % self.log_every_n_steps == 0:
            self.train_metrics.update(batch, logs)
        self.logger.experiment.add_scalar('train/tanh_softness', self.primal_block.tanh_softness, global_step = self.global_step)
        self.logger.experiment.add_scalar('train/learned_pert_influence', torch.abs(self.primal_block.learned_pert_influence), global_step = self.global_step)
        self.logger.experiment.add_scalar('train/mm_sign_influence', torch.abs(self.primal_block.mm_sign_influence), global_step = self.global_step)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx = 0):
        #try:
        if self.non_learned_updates_val:
            orig_batch = batch.clone()

        loss, batch_updated, logs = self.primal_rounds(batch, self.hparams.num_test_rounds, self.hparams.num_dual_iter_test, self.hparams.dual_improvement_slope_test, 0, is_training = False)
        data_name = self.val_datanames[dataset_idx]
        self.eval_metrics_val[data_name].update(batch_updated, logs)
        if self.non_learned_updates_val:
            _, batch_updated, logs = self.primal_rounds_non_learned(orig_batch, self.hparams.num_test_rounds, self.hparams.num_dual_iter_test, self.hparams.dual_improvement_slope_test)
            self.eval_metrics_val_non_learned[data_name].update(batch_updated, logs)
        # except:
        #     assert False, f'Error in validation step on files: {batch.file_path}'
        return loss

    def test_step(self, batch, batch_idx, dataset_idx = 0):
        assert len(batch.file_path) == 1, 'batch size 1 required for testing.'
        instance_name = os.path.basename(batch.file_path[0])
        data_name = self.test_datanames[dataset_idx]

        if self.non_learned_updates_test:
            orig_batch = batch.clone()
        loss, batch_updated, logs = self.primal_rounds(batch, self.hparams.num_test_rounds, self.hparams.num_dual_iter_test, self.hparams.dual_improvement_slope_test, grad_dual_itr_max_itr = 0)
        instance_level_metrics = PrimalMetrics(self.hparams.num_test_rounds, self.hparams.con_lp_features).to(batch.edge_index_var_con.device)
        instance_level_metrics.update(batch_updated, logs)
        self.log_metrics_test(instance_level_metrics, f'test_{data_name}_{instance_name}', 'learned')
        self.eval_metrics_test[data_name].update(batch_updated, logs)
        if self.non_learned_updates_test:
            loss, batch_updated, logs = self.primal_rounds_non_learned(orig_batch, self.hparams.num_test_rounds, self.hparams.num_dual_iter_test, self.hparams.dual_improvement_slope_test)
            instance_level_metrics = PrimalMetrics(self.hparams.num_test_rounds, self.hparams.con_lp_features, on_baseline=True).to(batch.edge_index_var_con.device)
            instance_level_metrics.update(batch_updated, logs)
            self.log_metrics_test(instance_level_metrics, f'test_{data_name}_{instance_name}', 'non_learned')
            self.eval_metrics_test_non_learned[data_name].update(batch_updated, logs)
        return loss