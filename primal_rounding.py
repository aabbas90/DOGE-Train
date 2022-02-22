import torch
from torch_scatter.scatter import scatter_add
from pytorch_lightning.core.lightning import LightningModule
from typing import List, Set, Dict, Tuple, Optional
from torch.optim import Adam
import time, tqdm, random, logging, os

from model.model import FeatureExtractor, PrimalPerturbationBlock
from model.solver_utils import init_solver_and_get_states, get_valid_edge_mask

class PrimalRoundingBDD(LightningModule):
    def __init__(self, 
                num_train_rounds: int,
                num_test_rounds: int,
                num_dual_iter_train: int,
                num_dual_iter_test: int,
                dual_improvement_slope_train: float,
                dual_improvement_slope_test: float,
                track_grad_after_iter: int,
                lr: float,
                loss_discount_factor:float,
                omega: float,
                var_lp_features: List[str],  # orig_obj, deg, obj pert
                con_lp_features: List[str],  # BDD lb, rhs, con type, degree
                edge_lp_features: List[str],  # lo cost, hi cost, mm diff, bdd sol, con coeff
                num_learned_var_f: int, 
                num_learned_con_f: int, 
                num_learned_edge_f: int,
                feature_extractor_depth: int,
                primal_predictor_depth: int,
                optimizer_name: str,
                datasets: List[str],
                test_fraction: List[int],
                test_datanames: Optional[List[str]] = None
                ):
        super(PrimalRoundingBDD, self).__init__()
        self.save_hyperparameters(
                'num_train_rounds',
                'num_test_rounds',
                'num_dual_iter_train',
                'num_dual_iter_test', 
                'dual_improvement_slope_train',
                'dual_improvement_slope_test',
                'track_grad_after_iter', 
                'lr',
                'omega',
                'loss_discount_factor',
                'var_lp_features',
                'con_lp_features',
                'edge_lp_features',
                'num_learned_var_f', 
                'num_learned_con_f', 
                'num_learned_edge_f',
                'feature_extractor_depth', 
                'primal_predictor_depth', 
                'optimizer_name',
                'datasets',
                'test_fraction',
                'test_datanames')

        self.lp_feature_extractor = FeatureExtractor(
                        num_var_lp_f = len(var_lp_features), out_var_dim = num_learned_var_f, 
                        num_con_lp_f = len(con_lp_features), out_con_dim = num_learned_con_f,
                        num_edge_lp_f = len(edge_lp_features), out_edge_dim = num_learned_edge_f,
                        depth = feature_extractor_depth)

        self.primal_block = PrimalPerturbationBlock(
                        num_var_lp_f = len(var_lp_features),
                        num_con_lp_f = len(con_lp_features), 
                        num_edge_lp_f = len(edge_lp_features),
                        depth = primal_predictor_depth,
                        var_dim = num_learned_var_f, 
                        con_dim = num_learned_con_f,
                        edge_dim = num_learned_edge_f)

        self.test_datanames = test_datanames
        self.console_logger = logging.getLogger('lightning')
        self.logger_interval = 20

    @classmethod
    def from_config(cls, cfg, test_datanames):
        return cls(
            num_train_rounds = cfg.TRAIN.NUM_ROUNDS,
            num_test_rounds = cfg.TEST.NUM_ROUNDS,
            var_lp_features = cfg.MODEL.VAR_LP_FEATURES,
            con_lp_features = cfg.MODEL.CON_LP_FEATURES,
            edge_lp_features = cfg.MODEL.EDGE_LP_FEATURES,
            num_dual_iter_train = cfg.TRAIN.NUM_DUAL_ITERATIONS,
            num_dual_iter_test = cfg.TEST.NUM_DUAL_ITERATIONS,
            dual_improvement_slope_train = cfg.TRAIN.DUAL_IMPROVEMENT_SLOPE,
            dual_improvement_slope_test = cfg.TEST.DUAL_IMPROVEMENT_SLOPE,
            track_grad_after_iter = cfg.TRAIN.TRACK_GRAD_AFTER_ITR,
            lr = cfg.TRAIN.BASE_LR,
            loss_discount_factor = cfg.TRAIN.LOSS_DISCOUNT_FACTOR,
            omega = cfg.MODEL.OMEGA,
            num_learned_var_f = cfg.MODEL.VAR_FEATURE_DIM, 
            num_learned_con_f = cfg.MODEL.CON_FEATURE_DIM,
            num_learned_edge_f = cfg.MODEL.EDGE_FEATURE_DIM,
            feature_extractor_depth = cfg.MODEL.FEATURE_EXTRACTOR_DEPTH,
            primal_predictor_depth = cfg.MODEL.PRIMAL_PRED_DEPTH,
            optimizer_name = cfg.TRAIN.OPTIMIZER,
            datasets = cfg.DATA.DATASETS,
            test_fraction = cfg.DATA.TEST_FRACTION,
            test_datanames = test_datanames)

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

    def configure_optimizers(self):
        if self.hparams.optimizer_name == 'Adam':
            return Adam(self.parameters(), lr=self.hparams.lr)
        else:
            raise ValueError(f'Optimizer {self.hparams.optimizer_name} not exposed.')

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch.edge_index_var_con = batch.edge_index_var_con.to(device)
        batch.omega = torch.tensor([self.hparams.omega], device = device)
        solvers, solver_state, per_bdd_sol, per_bdd_lb, dist_weights = init_solver_and_get_states(batch, device, 0, 1.0, batch.omega)

        batch.dist_weights = dist_weights # Isotropic weights.

        var_degree = scatter_add(torch.ones((batch.num_edges), device=device), batch.edge_index_var_con[0])
        var_degree[torch.cumsum(batch.num_vars, 0) - 1] = 0 # Terminal nodes, not corresponding to any primal variable.
        var_net_perturbation = torch.zeros_like(var_degree)
        batch.var_lp_f = torch.stack((batch.objective.to(device), var_degree, var_net_perturbation), 1) # Obj, Deg, Net. Pert
        batch.objective = None

        con_degree = scatter_add(torch.ones((batch.num_edges), device=device), batch.edge_index_var_con[1])
        batch.con_lp_f = torch.stack((per_bdd_lb, batch.rhs_vector.to(device), batch.con_type.to(device), con_degree), 1) # BDD lb, rhs, con type, degree
        batch.rhs_vector = None
        batch.con_type = None

        # Edge LP features:
        batch.edge_rest_lp_f = torch.stack((per_bdd_sol, batch.con_coeff.to(device)), 1)
        batch.solver_state = solver_state
        
        batch.solvers = solvers
        return batch

    # def on_train_start(self):
        # for (idx, data_name) in enumerate(self.test_datanames):
        #     self.logger.log_hyperparams(self.hparams, 
        #                             {f'hp/val_%_covered_{data_name}': 0, 
        #                             f'hp/val_lb_end_{data_name}': 0,
        #                             f'hp/gap_sum_{data_name}': 0})

    # def on_test_start(self):
    #     self.lb_evol_sum = None
    #     self.lb_evol_num = 0
    #     self.lb_evol_data_idx = None

    # def compute_training_start_itr(self):
    #     max_training_steps = self.num_training_steps
    #     current_step = self.global_step - 50
    #     assert current_step < max_training_steps
    #     max_training_steps_mod = max_training_steps // 10 # Divides into ten journeys.
    #     fraction = float(current_step) / max_training_steps_mod
    #     if fraction < 0:
    #         return 0
    #     fraction = fraction % 1
    #     fraction = fraction * fraction
    #     mean_start_step = fraction * (self.hparams.start_train_step_range[1] - self.hparams.start_train_step_range[0])
    #     proposed_start_step = np.round(np.random.normal(mean_start_step, 3)).astype(np.int32).item(0)
    #     return max(min(proposed_start_step, self.hparams.start_train_step_range[1]), self.hparams.start_train_step_range[0])

    def log_metrics(self, loss, losses, metrics, mode):
        self.console_logger.info(f'itr: {self.global_step}', end = '')
        for k, v in {**losses, **metrics}.items():
            self.log(mode + '_' + k, v)
            self.console_logger.info(f'{mode}_ + {k}: {v}', end = '')
        self.log(mode + '_loss', loss.item())

    # Find edges which correspond to valid primal variables (as each BDD contains 1 invalid primal variable each.)
    def get_valid_target_solution_edge(self, batch):
        valid_edge_mask = get_valid_edge_mask(batch)
        var_indices = batch.edge_index_var_con[0]
        valid_var_indices = var_indices[valid_edge_mask]
        gt_ilp_sol_var = torch.cat([torch.from_numpy(s).to(var_indices.device) for s in batch.gt_info['ilp_stats']['sol']], 0)
        gt_ilp_sol_edge = gt_ilp_sol_var[valid_var_indices]
        return gt_ilp_sol_edge, valid_edge_mask

    def primal_loss(self, batch, mm_pred, gt_ilp_sol_edge, valid_edge_mask, eps = 1e-3):
         # Gather the solution w.r.t all valid edges
        mm_pred_valid = mm_pred[valid_edge_mask]

        # if gt_ilp_solution > 0 then mm_pred should be < -eps and if gt_ilp_solution == 0 then mm_pred should be > eps:
        return torch.sum(torch.nn.ReLU()((gt_ilp_sol_edge * mm_pred_valid) + eps) + torch.nn.ReLU()(((gt_ilp_sol_edge - 1.0) * mm_pred_valid) + eps))

    def primal_accuracy(self, batch, mm_pred, gt_ilp_sol_edge, valid_edge_mask, eps = 1e-3):
        mm_pred_valid = mm_pred[valid_edge_mask]
        pred_f = (mm_pred_valid > eps).float()
        pred_t = (mm_pred_valid < -eps).float()
        tp = (gt_ilp_sol_edge * pred_t).sum()
        TPR = tp / gt_ilp_sol_edge.sum() 
        tn = ((1.0 - gt_ilp_sol_edge) * pred_f).sum()
        TNR = tn / (1.0 - gt_ilp_sol_edge).sum()
        return TPR.item(), TNR.item()

    def primal_rounds(self, batch, num_rounds, num_dual_iterations, improvement_slope, track_grads_from_step):
        assert(track_grads_from_step == 0) # Not fully ready yet.
        var_learned_f, con_learned_f, edge_learned_f = self.lp_feature_extractor(batch.var_lp_f, batch.con_lp_f, batch.solver_state, batch.edge_rest_lp_f, batch.edge_index_var_con)
        loss = 0
        losses_log = {}
        metrics_log = {}
        gt_ilp_sol_edge, valid_edge_mask = self.get_valid_target_solution_edge(batch)
        for r in range(num_rounds):
            solver_state, new_mm, new_var_lp_f = self.primal_block(batch.solvers, batch.var_lp_f, batch.con_lp_f, batch.solver_state, batch.edge_rest_lp_f, 
                                                                    var_learned_f, con_learned_f, edge_learned_f, 
                                                                    batch.dist_weights, batch.omega, batch.edge_index_var_con,
                                                                    num_dual_iterations, track_grads_from_step, improvement_slope)
            solver_state['def_mm'][~valid_edge_mask] = 0
            try:
                assert(torch.all(torch.isfinite(solver_state['def_mm'])))
            except:
                breakpoint()

            batch.solver_state = solver_state
            batch.var_lp_f = new_var_lp_f
            # Apply loss on new_mm with decay weight.
            current_loss = self.primal_loss(batch, new_mm, gt_ilp_sol_edge, valid_edge_mask)
            loss = loss + torch.pow(torch.tensor(self.hparams.loss_discount_factor), num_rounds - r - 1) * current_loss
            losses_log[f'loss_round_{r}'] = current_loss
            tpr, tnr = self.primal_accuracy(batch, new_mm, gt_ilp_sol_edge, valid_edge_mask)
            metrics_log[f'tpr_{r}'] = tpr
            metrics_log[f'tnr_{r}'] = tnr
        return loss, new_mm, losses_log, metrics_log

    def training_step(self, batch, batch_idx):
        if batch.gt_info is None:
            raise ValueError(f'GT should be known for training on: {batch.file_path}')
        
        loss, _, losses, metrics = self.primal_rounds(batch, self.hparams.num_train_rounds, self.hparams.num_dual_iter_train, self.hparams.dual_improvement_slope_train, 0)
        self.log_metrics(loss, losses, metrics, 'train')
        return loss 

    def validation_step(self, batch, batch_idx, dataset_idx = 0):
        if batch.gt_info is None:
            raise ValueError(f'GT should be known on: {batch.file_path}')
        
        loss, _, losses, metrics = self.primal_rounds(batch, self.hparams.num_test_rounds, self.hparams.num_dual_iter_test, self.hparams.dual_improvement_slope_test, 0)
        self.log_metrics(loss, losses, metrics, 'val')
        return loss

    def test_step(self, batch, batch_idx, dataset_idx = 0):
        if batch.gt_info is None:
            raise ValueError(f'GT should be known on: {batch.file_path}')
        
        loss, _, losses, metrics = self.primal_rounds(batch, self.hparams.num_test_rounds, self.hparams.num_dual_iter_test, self.hparams.dual_improvement_slope_test, 0)
        self.log_metrics(loss, losses, metrics, 'test')
        return loss
