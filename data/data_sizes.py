from cmath import nan
import os, pickle, time, argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir")
    args = parser.parse_args()
    print("Command Line Args:")
    print(args)
    root_dir = args.root_dir

    stats = []
    for path, subdirs, files in os.walk(root_dir):
        total_num_nodes = 0
        total_num_edges = 0
        total_num_instances = 0
        for instance_name in sorted(files):
            if not instance_name.endswith('.lp') or 'nan' in instance_name or 'normalized' in instance_name or 'slow_bdd' in instance_name or '_one_con' in instance_name or 'oom' in instance_name or 'too_easy' in instance_name:
                continue

            instance_path = os.path.join(path, instance_name)
            bdd_repr_path = instance_path.replace('.lp', '_bdd_repr.pkl')
            if not os.path.exists(bdd_repr_path):
                bdd_repr_path = instance_path.replace('.lp', '_bdd_repr_double.pkl')
            bdd_repr = pickle.load(open(bdd_repr_path, 'rb'))
            solver = pickle.loads(bdd_repr['solver_data'])

            rel_path = os.path.relpath(instance_path, root_dir)
            num_vars = bdd_repr['num_vars']
            num_cons = bdd_repr['num_cons']
            num_layers = bdd_repr['num_layers']
            num_nodes = num_vars + num_cons
            
            total_num_nodes += num_nodes
            total_num_edges += num_layers
            total_num_instances += 1.0
            print(f'{rel_path}, {num_vars}, {num_cons}, {num_nodes}, {num_layers}')
        # if total_num_instances > 0:
        #     print(f'{path}, {total_num_nodes / total_num_instances:.6}, {total_num_edges / total_num_instances:.6}')
