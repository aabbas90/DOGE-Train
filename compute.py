import torch
from torch_scatter.scatter import scatter_sum
from pytorch_lightning.core.lightning import LightningModule
from model.model import PrimalPerturbationBlock
from typing import List, Set, Dict, Tuple, Optional
from torch.optim import Adam
from collections import defaultdict
import random
import logging, os
import numpy as np
import time
import tqdm

class GNNBDDModel(LightningModule):
    def __init__(self, 
                num_iterations_train: int,
                num_iterations_test: int,
                lr: float,
                num_learned_var_channels: int, 
                num_learned_con_channels: int, 
                num_learned_edge_channels: int,
                feature_extractor_depth: int,
                bdd_layer_depth: int,
                optimizer_name: str,
                datasets: List[str],
                test_fraction: List[int],
                start_train_step_range: List[int], 
                bdd_exec_path: str = None,
                test_datanames: Optional[List[str]] = None
                ):
        super(GNNBDDModel, self).__init__()
        self.save_hyperparameters('num_iterations_train', 
                                'num_iterations_test', 
                                'lr', 
                                'num_learned_var_channels',
                                'num_learned_con_channels',
                                'num_learned_edge_channels',
                                'feature_extractor_depth', 
                                'bdd_layer_depth',
                                'backprop_mm_finite_diff',
                                'backprop_mm_finite_diff_step_size', 
                                'optimizer_name',
                                'datasets',
                                'test_fraction',
                                'start_train_step_range',
                                'fixed_feature_indices'
                                )
        self.fixed_feature_indices = fixed_feature_indices
        self.test_datanames = test_datanames
        self.lp_feature_extractor = FeatureExtractor(
                                num_learned_var_channels, 
                                num_learned_con_channels, 
                                num_learned_edge_channels, 
                                feature_extractor_depth)

        self.bdd_layer = BDDLayer(num_learned_var_channels, 
                                num_learned_con_channels, 
                                num_learned_edge_channels, 
                                fixed_feature_indices,
                                bdd_layer_depth,
                                backprop_mm_finite_diff, 
                                backprop_mm_finite_diff_step_size)

        self.learned_updates_test = True # False will do hand-crafted update using mma.
        self.run_bdd_seq_test = False # True will run bdd sequential solver from command line
        self.console_logger = logging.getLogger('lightning')
        self.logger_interval = 20
        self.lb_evol_sum = None
        self.lb_evol_num = 0
        self.lb_evol_data_idx = None
        self.bdd_exec_path = bdd_exec_path

    @classmethod
    def from_config(cls, cfg, fixed_feature_indices, test_datanames):
        return cls(
            num_learned_var_channels = cfg.MODEL.VAR_FEATURE_DIM, 
            num_learned_con_channels = cfg.MODEL.CON_FEATURE_DIM,
            num_learned_edge_channels = cfg.MODEL.EDGE_FEATURE_DIM,
            feature_extractor_depth = cfg.MODEL.FEATURE_EXTRACTOR_DEPTH,
            bdd_layer_depth = cfg.MODEL.BDD_LAYER_DEPTH,
            fixed_feature_indices = fixed_feature_indices,
            backprop_mm_finite_diff = cfg.TRAIN.BACKPROP_MM_FINITE_DIFF, 
            backprop_mm_finite_diff_step_size = cfg.TRAIN.BACKPROP_MM_FINITE_DIFF_STEP_SIZE,
            lr = cfg.TRAIN.BASE_LR,
            num_iterations_train = cfg.TRAIN.NUM_ITERATIONS,
            num_iterations_test = cfg.TEST.NUM_ITERATIONS,
            optimizer_name = cfg.TRAIN.OPTIMIZER,
            datasets = cfg.DATA.DATASETS,
            test_fraction = cfg.DATA.TEST_FRACTION,
            start_train_step_range = cfg.TRAIN.START_STEP_RANGE,
            test_datanames = test_datanames,
            bdd_exec_path = cfg.TEST.BDD_SOLVER_EXEC
        )        

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

    def transfer_batch_to_device(self, batch, device):
        batch.var_fixed_f = batch.var_fixed_f.to(device)
        batch.con_fixed_f = batch.con_fixed_f.to(device)
        batch.edge_fixed_f = batch.edge_fixed_f.to(device)
        batch.edge_index_var_con_batch = batch.edge_index_var_con_batch.to(device)
        batch.con_fixed_f_batch = batch.con_fixed_f_batch.to(device)
        batch.edge_index_var_con = batch.edge_index_var_con.to(device)
        if batch.gt_obj is not None:
            batch.gt_obj = batch.gt_obj.to(device)
        return batch

    # def on_train_start(self):
        # for (idx, data_name) in enumerate(self.test_datanames):
        #     self.logger.log_hyperparams(self.hparams, 
        #                             {f'hp/val_%_covered_{data_name}': 0, 
        #                             f'hp/val_lb_end_{data_name}': 0,
        #                             f'hp/gap_sum_{data_name}': 0})

    def on_test_start(self):
        self.lb_evol_sum = None
        self.lb_evol_num = 0
        self.lb_evol_data_idx = None

    def compute_training_start_itr(self):
        max_training_steps = self.num_training_steps
        current_step = self.global_step - 50
        assert current_step < max_training_steps
        max_training_steps_mod = max_training_steps // 10 # Divides into ten journeys.
        fraction = float(current_step) / max_training_steps_mod
        if fraction < 0:
            return 0
        fraction = fraction % 1
        fraction = fraction * fraction
        mean_start_step = fraction * (self.hparams.start_train_step_range[1] - self.hparams.start_train_step_range[0])
        proposed_start_step = np.round(np.random.normal(mean_start_step, 3)).astype(np.int32).item(0)
        return max(min(proposed_start_step, self.hparams.start_train_step_range[1]), self.hparams.start_train_step_range[0])

    def training_step(self, batch, batch_idx):
        if batch.gt_obj is None:
            raise ValueError(f'GT obj value should be known for training on: {batch.file_path}')
        
        start_step_training = self.compute_training_start_itr()
        self.log('start_step_training', start_step_training)
        batch = self.compute_lp_features(batch)
        _, losses, metrics = self.run_solver(batch = batch, 
                                            num_iterations = start_step_training + self.hparams.num_iterations_train, 
                                            track_grads_from_step = start_step_training,
                                            return_lbs = False, 
                                            run_mma = False)

        self.console_logger.info(f'itr: {self.global_step}', end = '')
        for k, v in {**losses, **metrics}.items():
            self.log('train_' + k, v)
            self.console_logger.info(f'train_ + {k}: {v}', end = '')
        loss = sum(losses.values())
        self.log('train_loss', loss)
        return loss 

    def on_test_end(self):
        self.log_lb_curve()

    def accumulate_lb_evol(self, lb_evol, batch_size, dataset_idx, identifier):
        if self.lb_evol_sum is not None:
            if self.lb_evol_data_idx == dataset_idx:
                self.lb_evol_sum = self.lb_evol_sum + lb_evol
                self.lb_evol_num = self.lb_evol_num + batch_size
                self.lb_evol_identifier = identifier
            else:
                self.log_lb_curve()
        if self.lb_evol_sum is None:
            self.lb_evol_sum = lb_evol
            self.lb_evol_num = batch_size
            self.lb_evol_data_idx = dataset_idx
            self.lb_evol_identifier = identifier

    def log_lb_curve(self):
        for s in range(self.lb_evol_sum.size):
            self.logger.experiment.add_scalars(f'{self.test_datanames[self.lb_evol_data_idx]}_lb_end', 
                {self.lb_evol_identifier: self.lb_evol_sum[s] / self.lb_evol_num}, 
                s * self.logger_interval)
        self.lb_evol_sum = None

    def test_step(self, batch, batch_idx, dataset_idx = 0):
        data_name = self.test_datanames[dataset_idx]
        if self.run_bdd_seq_test:
            return self.eval_sequential_solver(batch, data_name, dataset_idx)

        return self.eval_internal(batch, data_name, dataset_idx, 'test', self.learned_updates_test)

    def validation_step(self, batch, batch_idx, dataset_idx = 0):
        data_name = self.test_datanames[dataset_idx]
        return self.eval_internal(batch, data_name, dataset_idx, 'val', True)

    def eval_internal(self, batch, dataset_name, dataset_idx, mode, learned_updates):
        rounds_done_already = 0
        lb_0 = None
        batch_size = len(batch.bdd_solver)
        identifier = 'learned' if learned_updates else 'mma'
        start_time = time.time()
        batch = self.compute_lp_features(batch)
        gap_sum = 0
        gap_initial = 0
        prev_lb = None
        lb_evol = []
        for r in range(self.hparams.num_iterations_test):
            batch, losses, _, lb_end, lb_initial = self.run_solver(batch = batch, num_iterations = 1, track_grads_from_step = 1, return_lbs = True, run_mma = not learned_updates)

            if lb_0 is None:
                lb_0 = lb_initial
    
            if r % self.logger_interval == 0:
                lb_evol.append(lb_end.sum().item())
    
            prev_lb = lb_end
            gap_sum += ((batch.gt_obj - lb_end) / (batch.gt_obj - lb_initial))

        # Loss on last round, as a single measure of final performance:
        if batch.gt_obj is not None:
            cov = self.compute_relative_increase(lb_end, lb_0, batch.gt_obj, lb_0)
            self.log(f'{mode}_{dataset_name}_gap_sum/{identifier}', gap_sum, add_dataloader_idx = False)
            self.log(f'{mode}_{dataset_name}_percent_covered/{identifier}', cov * 100.0, add_dataloader_idx = False)
            self.log(f'{mode}_{dataset_name}_avg_primal_cost/{identifier}', batch.gt_obj, add_dataloader_idx = False)

        self.log(f'{mode}_{dataset_name}_lb_end/{identifier}', lb_end, add_dataloader_idx = False)
        self.log(f'{mode}_{dataset_name}_lb_start/{identifier}', lb_0, add_dataloader_idx = False)
        if 'test' in mode:
            identifier = 'test_' + identifier
        else:
            identifier = 'val_' + str(self.global_step)
        self.accumulate_lb_evol(np.stack(lb_evol), lb_end.size, dataset_idx, identifier)
        return {}

    def eval_sequential_solver(self, batch, dataset_name, dataset_idx):
        lb_evol = []

        for path in tqdm.tqdm(batch.file_path):
            try:
                lbs = run_bdd_sequential_solver(self.bdd_exec_path, path, 2 * self.hparams.num_iterations_test)
                lb_evol = lbs[::self.logger_interval]
                self.accumulate_lb_evol(np.stack(lb_evol), 1, dataset_idx, 'test_bdd_seq')
            except:
                print(f"BDD Solve: Error in {path}")
        
    def compute_lp_features(self, batch):
        con_rhs_index = self.fixed_feature_indices['con'].get_index('rhs')
        con_leq_index = self.fixed_feature_indices['con'].get_index('leq')
        con_geq_index = self.fixed_feature_indices['con'].get_index('geq')
        con_deg_index = self.fixed_feature_indices['con'].get_index('deg')

        var_in_features = batch.var_fixed_f[:, [self.fixed_feature_indices['var'].get_index('obj'), self.fixed_feature_indices['var'].get_index('deg')]]
        con_in_features = batch.con_fixed_f[:, [con_rhs_index, con_leq_index, con_geq_index, con_deg_index]]
        edge_in_features = batch.edge_fixed_f[:, [self.fixed_feature_indices['edge'].get_index('coeff')]]
        batch.var_learned_f, batch.con_learned_f, batch.edge_learned_f = self.lp_feature_extractor(
            var_in_features, con_in_features, edge_in_features, 
            batch.edge_index_var_con_batch, batch.edge_index_var_con, batch.con_fixed_f_batch)

        return batch

    def check_nans(self, batch):
        var_finite = torch.all(torch.isfinite(batch.var_learned_f)) and torch.all(torch.isfinite(batch.var_fixed_f))
        con_finite = torch.all(torch.isfinite(batch.con_learned_f)) and torch.all(torch.isfinite(batch.con_fixed_f))
        edge_finite = torch.all(torch.isfinite(batch.edge_learned_f)) and torch.all(torch.isfinite(batch.edge_fixed_f))
        error_desc = ''
        if not var_finite:
            error_desc = ' var features '
        if not con_finite:
            error_desc += ' con features '
        if not edge_finite:
            error_desc += ' edge features '
        if not (var_finite and con_finite and edge_finite):
            raise ValueError(f'NaNs found in: {error_desc}, \n files: {", ".join(batch.file_path)}')

    def run_solver(self, batch, num_iterations, track_grads_from_step = 0, return_lbs = False, run_mma = False):
        with torch.no_grad():
            _, initial_lb = self.compute_metrics(batch.con_fixed_f, batch.con_fixed_f_batch, batch.gt_obj)
            prev_lb = initial_lb

        sum_rel_increase = 0
        count = 0
        track_grads = False
        for r in range(num_iterations):
            if self.training and (not track_grads) and (r >= track_grads_from_step):
                with torch.set_grad_enabled(True):
                    batch = self.compute_lp_features(batch)
                    self.check_nans(batch)
                track_grads = True
            with torch.set_grad_enabled(track_grads):
                batch.var_fixed_f, batch.con_fixed_f, batch.edge_fixed_f = self.bdd_layer(batch.var_learned_f, batch.var_fixed_f,
                                                                                        batch.con_learned_f, batch.con_fixed_f,
                                                                                        batch.edge_learned_f, batch.edge_fixed_f,
                                                                                        batch.edge_index_var_con_batch, batch.edge_index_var_con, 
                                                                                        batch.con_fixed_f_batch, batch.bdd_solver, 
                                                                                        run_mma)
                self.check_nans(batch)

                rel_increase, lb_end = self.compute_metrics(batch.con_fixed_f, batch.con_fixed_f_batch, batch.gt_obj, prev_lb, initial_lb)
                prev_lb = lb_end.detach()
                if (not self.training or track_grads) and batch.gt_obj is not None:
                    sum_rel_increase = sum_rel_increase + rel_increase
                    count = count + 1

        if rel_increase is not None and torch.any(~torch.isfinite(rel_increase)):
            breakpoint()
            raise ValueError('NaNs found in rel_increase')
        
        losses = {}
        metrics = {'average_lb_increase': (lb_end - initial_lb).mean()}
        if batch.gt_obj is not None:
            losses['loss_neg_rel_increase'] =  -1.0 * sum_rel_increase / count
            metrics.update({'rel_increase': rel_increase, 'avg_primal_cost': batch.gt_obj.mean()})

        if return_lbs:
            return batch, losses, metrics, lb_end, initial_lb
        else:
            return batch, losses, metrics

    def compute_metrics(self, con_fixed_f, con_batch_index, gt_obj, prev_lb_batch = None, initial_lb_batch = None):
        '''
        prev_lb_batch: lower bound before the current forward pass.
        initial_lb_batch: lower bound after the current training round. So initial_lb_batch < prev_lb_batch (if we are always making improvement)
        '''

        lb_per_bdd = con_fixed_f[:, self.fixed_feature_indices['con'].get_index('lb')]
        new_lb_per_batch = scatter_sum(lb_per_bdd, con_batch_index)
        rel_increase = None
        if prev_lb_batch is not None and gt_obj is not None:
            assert(prev_lb_batch.requires_grad == False)
            rel_increase = self.compute_relative_increase(new_lb_per_batch, prev_lb_batch, gt_obj, initial_lb_batch)
        return rel_increase, new_lb_per_batch

    def compute_relative_increase(self, new_lb_per_batch, prev_lb_batch, gt_obj, initial_lb_batch):
        return torch.mean((new_lb_per_batch - prev_lb_batch) / (gt_obj - torch.minimum(initial_lb_batch, prev_lb_batch)  + 1e-5))
        # return torch.mean((new_lb_per_batch - prev_lb_batch) / (gt_obj - initial_lb_batch  + 1e-5))