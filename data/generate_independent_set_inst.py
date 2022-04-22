import ecole, os
from tqdm import tqdm
import gurobipy as gp

num_instances_train = 240
n_nodes_train = 10000
edge_probability_train = 0.25
affinity_train = 4

num_instances_test = 60
n_nodes_test = 50000
edge_probability_test = 0.25
affinity_test = 8

seed = 1311

def CreateIndependentSetGenerator(n_nodes, edge_probability, affinity):
    return ecole.instance.IndependentSetGenerator(n_nodes = n_nodes, edge_probability = edge_probability, affinity = affinity)

# random_instance_generator = CreateIndependentSetGenerator(n_nodes_test, edge_probability_test, affinity_test)
# # random_instance_generator = CreateIndependentSetGenerator(n_nodes_train, edge_probability_train, affinity_train)
# random_instance_generator.seed(seed)

# for n in tqdm(range(num_instances_test)):
#     #out_path = f'/home/ahabbas/data/learnDBCA/independent_set_random/test_split/instances/{num_instances_test - 1 + n}.lp'
#     out_path = f'/home/ahabbas/data/learnDBCA/independent_set_random/test_split/instances/{n}.lp'
#     if not os.path.exists(out_path):
#         ilp_ecole = next(random_instance_generator)
#         ilp_scip = ilp_ecole.as_pyscipopt()
#         #ilp_scip.writeProblem(f'/home/ahabbas/data/learnDBCA/independent_set_random/train_split/instances/{n}.lp')
#         ilp_scip.writeProblem(out_path)


root_dir = '/home/ahabbas/data/learnDBCA/independent_set_random/test_split/instances/'
for name in os.listdir(root_dir):
    in_path = os.path.join(root_dir, name)
    out_path = in_path
    ilp_gurobi = gp.read(in_path)
    ilp_gurobi.setParam('Presolve', 1) # conservative.
    ilp_gurobi = ilp_gurobi.presolve()
    ilp_gurobi.update()
    num_binary = ilp_gurobi.getAttr('NumBinVars')
    num_vars = ilp_gurobi.getAttr('NumVars')
    assert(num_binary == num_vars)
    ilp_gurobi.write(out_path)