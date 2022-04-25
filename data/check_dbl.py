import pickle, os, tqdm, argparse

def solve_dual_bdd(bdd_repr, improvement_slope, num_iterations, omega):
    solver = pickle.loads(bdd_repr['solver_data'])
    print(f'Initial lb: {solver.lower_bound()}')
    solver.non_learned_iterations(omega, num_iterations, improvement_slope)
    solver.distribute_delta() # make deferred min-marginals zero.
    print(f'Final lb: {solver.lower_bound()}')
    bdd_repr['solver_data'] = pickle.dumps(solver, -1) # Overwrite bdd representation with update costs.
    return bdd_repr

def generate_ilps(root_dir, suffix, improvement_slope, num_iterations, omega):
    file_list = []
    for path, directories, files in os.walk(root_dir):
        for file in files:
            if not file.endswith('.pkl') or not 'double' in file or not 'scr20' in file or 'conv' in file:
                continue
            file_list.append([path, file])

    for bdd_rep_path, filename in tqdm.tqdm(file_list):
        ext = os.path.splitext(filename)[1]
        out_filename = filename.replace(ext, suffix + ext)
        bdd_repr = pickle.load(open(os.path.join(bdd_rep_path, filename), 'rb'))
        solve_dual_bdd(bdd_repr, improvement_slope, num_iterations, omega)
        breakpoint()

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to config root dir containing .lp files in any child dirs.")
parser.add_argument("--num_iterations", default=50000)
parser.add_argument("--improvement_slope", default=0.0)
parser.add_argument("--omega", default=0.5)
args = parser.parse_args()

root_dir = args.input_dir
generate_ilps(root_dir, '', args.improvement_slope, args.num_iterations, args.omega)
