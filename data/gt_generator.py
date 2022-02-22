import gurobipy as gp
import time

def get_solution(model):
    vars = model.getVars()
    solution = {}
    obj_value = 0.0
    for var in vars:
        solution[var.VarName] = var.X
        obj_value += var.X * var.Obj # Does not account constant term in objective vector same as BDD solver.
    return solution, obj_value

def presolve_and_generate_gt(ilp_path):
    """Given an ILP first presolve it and see if all variables remain binary otherwise return None, None. If all binary then generate the following using gurobi:
    1. LP relaxation solution, objective and time need to produce it.
    2. original ILP solution, objective and time need to produce it. """
    ilp_gurobi = gp.read(ilp_path)
    ilp_gurobi = ilp_gurobi.presolve()
    num_binary = ilp_gurobi.getAttr('NumBinVars')
    num_vars = ilp_gurobi.getAttr('NumVars')
    if num_binary != num_vars:
        return None, None
    ilp_gurobi.write(ilp_path) # Overwrite with presolved instance.
    lp_relaxation = ilp_gurobi.relax()
    lp_relaxation.Params.Method = 1 # Dual simplex.
    start_time = time.time()
    lp_relaxation.optimize()
    lp_relaxation_time = time.time() - start_time
    lp_solution, lp_obj_value = get_solution(lp_relaxation)
    lp_stats = {'time': lp_relaxation_time, 'obj': lp_obj_value, 'sol': lp_solution}
    start_time = time.time()
    ilp_gurobi.optimize()
    ilp_time = time.time() - start_time
    ilp_solution, ilp_obj_value = get_solution(ilp_gurobi)
    ilp_stats = {'time': ilp_time, 'obj': ilp_obj_value, 'sol': ilp_solution}
    return lp_stats, ilp_stats


# generate_gt('/home/ahabbas/data/learnDBCA/random_instances_with_lp/IndependentSet_n_nodes_500_edge_probability_0.25_affinity_4_seed_1/ilp_instance_0.lp')