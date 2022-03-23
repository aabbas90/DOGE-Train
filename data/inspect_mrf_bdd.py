import pickle
import time
import torch

bdd_repr = pickle.load(open('/home/ahabbas/data/learnDBCA/cv_structure_pred/mrf/protein_folding/1CKK_bdd_repr.pkl', 'rb'))
solver = pickle.loads(bdd_repr['solver_data'])

num_itr = 5
st = time.time()

for i in range(num_itr):
    solver.non_learned_iterations(0.5, 1, 1e-6)

en = time.time()
print(f'Def MMA time: {en - st}')

mm_out = torch.empty((solver.nr_layers()), device = 'cuda', dtype = torch.float32)
lb_out = torch.empty((solver.nr_bdds()), device = 'cuda', dtype = torch.float32)

st = time.time()
for i in range(num_itr):
    solver.flush_solver_states()
    solver.all_min_marginal_differences(mm_out.data_ptr())
    solver.lower_bound_per_bdd(lb_out.data_ptr())
en = time.time()
print(f'Full coord time: {en - st}')
