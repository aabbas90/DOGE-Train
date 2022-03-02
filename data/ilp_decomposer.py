import gurobipy as gp
from torch_geometric.utils import from_scipy_sparse_matrix, to_networkx, to_undirected
from torch_geometric.data import Data
import numpy as np
import networkx as nx
import time, pickle, tqdm
import os.path
import argparse
from gt_generator import presolve_and_generate_gt

def sample_ilp(ilp_gurobi, graph_nx, num_vars, num_cons, max_num_constraints, max_num_vars):
    restricted_ilp = ilp_gurobi.copy()
    if num_cons > max_num_constraints and num_vars > max_num_vars:
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
        all_cons = restricted_ilp.getConstrs()
        for con in range(num_cons):
            if picked_constraints[con]:
                continue
            else:
                cons_to_remove.append(all_cons[con])
        restricted_ilp.remove(cons_to_remove)
        restricted_ilp.update()
    else:
        current_num_vars = num_vars
        current_num_cons = num_cons
    restricted_ilp = restricted_ilp.presolve()
    restricted_ilp.setAttr('ObjCon', 0.0)
    restricted_ilp.update()
    print(f'current_num_vars: {current_num_vars}, current_num_cons: {current_num_cons}')
    print('picked vars. # {} / {}, picked cons. # {} / {}'.format(restricted_ilp.getAttr('NumVars'), num_vars, restricted_ilp.getAttr('NumConstrs'), num_cons))
    return restricted_ilp

def decompose_ilp(ilp_path, out_dir, max_num_constraints, max_num_vars, max_nr_decomp):
    ilp_gurobi = gp.read(ilp_path)
    num_binary = ilp_gurobi.getAttr('NumBinVars')
    num_vars = ilp_gurobi.getAttr('NumVars')
    num_cons = ilp_gurobi.getAttr('NumConstrs')
    nr_decompositions = min(int(0.5 * num_cons / max_num_constraints) + 1, max_nr_decomp)
    A = ilp_gurobi.getA()
    edge_indices, _ = from_scipy_sparse_matrix(A)
    # Increase node indices of constraints since it is a bipartite graph. So 0:num_vars correspond to variable nodes and rest num_vars: to constraints
    edge_indices[0, :] += num_vars
    edge_indices_undir = to_undirected(edge_indices)
    graph_nx = None
    for d in range(nr_decompositions):
        lp_folder = os.path.join(out_dir, 'instances') 
        lp_path = lp_folder + '/' + str(d) + '.lp'
        os.makedirs(lp_folder, exist_ok = True)
        if graph_nx is None:
            graph_nx = to_networkx(Data(edge_index=edge_indices_undir, num_nodes = num_vars + num_cons), to_undirected=True)

        restricted_ilp = sample_ilp(ilp_gurobi, graph_nx, num_vars, num_cons, max_num_constraints, max_num_vars)
        restricted_ilp.write(lp_path)

def generate_ilps(root_dir, out_root_dir, max_num_constraints, max_num_vars, max_nr_decomp):
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
        decompose_ilp(lp_path, out_folder, max_num_constraints, max_num_vars, max_nr_decomp)

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", help="path to config root dir containing .lp files in any child dirs.")
parser.add_argument("output_dir")
args = parser.parse_args()

root_dir = args.input_dir
out_dir = args.output_dir
print(f'root_dir: {root_dir}')
print(f'out_dir: {out_dir}')
# generate_ilps(root_dir, out_dir, 10000, 50000, 5) # For GM
generate_ilps(root_dir, out_dir, 10000, 50000, 50) # For CT