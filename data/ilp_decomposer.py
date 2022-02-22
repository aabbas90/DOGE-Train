import gurobipy as gp
from torch_geometric.utils import from_scipy_sparse_matrix, to_networkx, to_undirected
from torch_geometric.data import Data
import numpy as np
import networkx as nx
import time, pickle, tqdm
import os.path
import argparse


def sample_ilp(ilp_gurobi, graph_nx, num_vars, num_cons, max_num_constraints, max_num_vars):
    starting_con = np.random.randint(num_vars, num_vars + num_cons)
    shortest_path_lengths = nx.shortest_path_length(graph_nx, source = starting_con)
    current_num_cons = 0
    current_num_vars = 0
    picked_constraints = np.zeros((num_cons), dtype=np.byte)
    max_dist = np.inf
    stop_adding_constraints = False
    for node in sorted(shortest_path_lengths, key = shortest_path_lengths.get):
        dist = shortest_path_lengths[node]
        if dist % 2: # Odd. So should be a variable node.
            assert node < num_vars
            current_num_vars += 1 
        else:
            assert node >= num_vars
            if stop_adding_constraints:
                if dist >= max_dist:
                    break
                continue
            picked_constraints[node - num_vars] = 1
            current_num_cons += 1
            if current_num_cons > max_num_constraints or current_num_vars > max_num_vars:
                max_dist = dist + 2 # No further constraints will be added but variables can still be added having distance dist + 1.
                stop_adding_constraints = True
    cons_to_remove = []
    restricted_ilp = ilp_gurobi.copy()
    all_cons = restricted_ilp.getConstrs()
    for con in range(num_cons):
        if picked_constraints[con]:
            continue
        else:
            cons_to_remove.append(all_cons[con])
    restricted_ilp.remove(cons_to_remove)
    restricted_ilp.update()
    restricted_ilp = restricted_ilp.presolve()
    restricted_ilp.setAttr('ObjCon', 0.0)
    restricted_ilp.update()
    print(f'current_num_vars: {current_num_vars}, current_num_cons: {current_num_cons}')
    print('picked vars. # {} / {}, picked cons. # {} / {}'.format(restricted_ilp.getAttr('NumVars'), num_vars, restricted_ilp.getAttr('NumConstrs'), num_cons))
    return restricted_ilp

def get_solution(model):
    vars = model.getVars()
    solution = {}
    obj_value = 0.0
    for var in vars:
        solution[var.VarName] = var.X
        obj_value += var.X * var.Obj # Does not account constant term in objective vector same as BDD solver.
    return solution, obj_value

def solve_ilp(ilp_gurobi):
    num_binary = ilp_gurobi.getAttr('NumBinVars')
    num_vars = ilp_gurobi.getAttr('NumVars')
    # if num_binary != num_vars:
    #     return None, None

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

def decompose_ilp(ilp_path, out_dir, max_num_constraints, max_num_vars, nr_decompositions):
    assert nr_decompositions > 1
    ilp_gurobi = gp.read(ilp_path)
    num_binary = ilp_gurobi.getAttr('NumBinVars')
    num_vars = ilp_gurobi.getAttr('NumVars')
    num_cons = ilp_gurobi.getAttr('NumConstrs')
    if max_num_vars >= num_vars or max_num_constraints >= num_cons: # Not pure binary ilp, or decomposition not possible.
        print(f'num_vars: {num_vars}, num_cons: {num_cons}')
        breakpoint() #flywing_11.lp, DIC-C2DH-HeLA, Fluo-C2DL-MSC_01.lp, Fluo-C2DL-MSC_02.lp, PhC-C2DH-U373_01.lp, PhC-C2DH-U373_02.lp, Fluo-N2DH-GOWT1_02, Fluo-N2DH-GOWT1_01, 
        return

    A = ilp_gurobi.getA()
    edge_indices, _ = from_scipy_sparse_matrix(A)
    # Increase node indices of constraints since it is a bipartite graph. So 0:num_vars correspond to variable nodes and rest num_vars: to constraints
    edge_indices[0, :] += num_vars
    edge_indices_undir = to_undirected(edge_indices)
    graph_nx = to_networkx(Data(edge_index=edge_indices_undir, num_nodes = num_vars + num_cons), to_undirected=True)
    for d in range(nr_decompositions):
        lp_folder = os.path.join(out_dir, 'instances') 
        lp_path = lp_folder + '/' + str(d) + '.lp'
        os.makedirs(lp_folder, exist_ok = True)
        sol_folder = os.path.join(out_dir, 'solutions')
        sol_path = sol_folder + '/' + str(d) + '.pkl'
        os.makedirs(sol_folder, exist_ok = True)

        restricted_ilp = sample_ilp(ilp_gurobi, graph_nx, num_vars, num_cons, max_num_constraints, max_num_vars)
        lp_stats, ilp_stats = solve_ilp(restricted_ilp)
        if lp_stats is None:
            continue

        restricted_ilp.write(lp_path)
        gt_info = {"lp_stats": lp_stats, "ilp_stats": ilp_stats}
        pickle.dump(gt_info, open(sol_path, "wb"))
        print(f'Wrote file: {lp_path} with solution: {sol_path}\n\n')

def generate_ilps(root_dir, out_root_dir, max_num_constraints, max_num_vars, nr_decompositions):
    file_list = []
    for path, directories, files in os.walk(root_dir):
        for file in files:
            if not file.endswith('.lp'):
                continue
            lp_path = os.path.join(path, file)
            file_list.append(lp_path)

    for lp_path in tqdm.tqdm(file_list):
        rel_path = os.path.relpath(lp_path, root_dir)
        filename = os.path.basename(lp_path)
        out_folder = os.path.join(out_root_dir, os.path.splitext(rel_path)[0])
        os.makedirs(out_folder, exist_ok = True)
        decompose_ilp(lp_path, out_folder, max_num_constraints, max_num_vars, nr_decompositions)

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", help="path to config root dir containing .lp files in any child dirs.")
parser.add_argument("output_dir")
args = parser.parse_args()

root_dir = args.input_dir
out_dir = args.output_dir
print(f'root_dir: {root_dir}')
print(f'out_dir: {out_dir}')
generate_ilps(root_dir, out_dir, 10000, 50000, 5) # For GM
# generate_ilps(root_dir, out_dir, 50000, 250000, 20) # For CT
#generate_ilps('/home/ahabbas/data/cell-tracking-AISTATS-2020/', '/home/ahabbas/data/learnDBCA/cv_structure_pred/cell-tracking-AISTATS-2020/', 10000, 100000, 20)
#generate_ilps('/BS/discrete_opt/work/datasets/graph_matching/', '/home/ahabbas/data/learnDBCA/cv_structure_pred/graph_matching/', 10000, 100000, 20)