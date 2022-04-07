import pickle
import time
import torch

bdd_repr = pickle.load(open('/home/ahabbas/data/learnDBCA/cv_structure_pred/graph-matching/worms/train_split/worm01-16-03-11-1745/instances/0_bdd_repr_dual_converged.pkl', 'rb'))
solver = pickle.loads(bdd_repr['solver_data'])
mm_out = torch.empty((solver.nr_layers()), device = 'cuda', dtype = torch.float32)
solver.all_min_marginal_differences(mm_out.data_ptr())
print(f'mm_out: min: {mm_out.min()}, max: {mm_out.max()}')
assert(torch.all(torch.isfinite(mm_out)))
breakpoint()
