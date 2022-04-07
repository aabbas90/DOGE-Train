import gurobipy as gp



ilp_path = '/home/ahabbas/data/learnDBCA/cv_structure_pred/graph-matching/worms/train_split/worm01-16-03-11-1745/instances/0.lp'
ilp_gurobi = gp.read(ilp_path)
num_binary = ilp_gurobi.getAttr('NumBinVars')
num_int_vars = ilp_gurobi.getAttr('NumIntVars') - num_binary
assert num_int_vars == 0, f'num_int_vars: {num_int_vars}'
num_vars = ilp_gurobi.getAttr('NumVars')

presolved_ilp = ilp_gurobi.presolve()
num_binary = presolved_ilp.getAttr('NumBinVars')
num_vars = presolved_ilp.getAttr('NumVars')
num_int_vars = ilp_gurobi.getAttr('NumIntVars') - num_binary
assert num_int_vars == 0, f'num_int_vars: {num_int_vars}'
breakpoint()