import os
import torch_geometric
import pickle
from data.ilp_converters import create_bdd_repr_from_ilp, create_graph_from_bdd_repr, solve_dual_bdd
from data.gt_generator import generate_gt_gurobi

class ILPDiskDataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, data_root_dir, files_to_load, read_dual_converged, need_gt):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.data_root_dir = data_root_dir
        self.files_to_load = files_to_load
        self.read_dual_converged = read_dual_converged
        self.need_gt = need_gt
        self.process_custom()

    @classmethod
    def from_config(cls, cfg, data_name):
        params = cfg.DATA[data_name + '_PARAMS']
        data_root_dir = params.root_dir
        files_to_load = params.files_to_load
        read_dual_converged = params.read_dual_converged
        need_gt = True
        if 'need_gt' in params:
            need_gt = params.need_gt
        return cls(
            data_root_dir = data_root_dir,
            files_to_load = files_to_load,
            read_dual_converged = read_dual_converged,
            need_gt = need_gt)

    def process_custom(self):
        self.file_list = []
        for path, subdirs, files in os.walk(self.data_root_dir):
            for instance_name in files:
                if not instance_name.endswith('.lp') or 'nan' in instance_name or 'normalized' in instance_name:
                    continue

                if len(self.files_to_load) > 0 and instance_name not in self.files_to_load:
                    continue

                instance_path = os.path.join(path, instance_name)
                sol_path = os.path.join(path.replace('instances', 'solutions'), 
                                        instance_name.replace('.lp', '.pkl'))
                
                if not os.path.exists(sol_path):
                    if self.need_gt:
                        lp_stats, ilp_stats = generate_gt_gurobi(instance_path)
                        gt_info = {"lp_stats": lp_stats, "ilp_stats": ilp_stats}
                        pickle.dump(gt_info, open(sol_path, "wb"))
                    else:
                        empty_sol = {'time': None, 'obj': None, 'sol_dict': None, 'sol': None}
                        gt_info = {"lp_stats": empty_sol, "ilp_stats": empty_sol}
                        pickle.dump(gt_info, open(sol_path, "wb"))
                else:
                    gt_info = pickle.load(open(sol_path, 'rb'))

                bdd_repr_path = instance_path.replace('.lp', '_bdd_repr.pkl')
                bdd_repr_conv_path = instance_path.replace('.lp', '_bdd_repr_dual_converged.pkl')

                if not os.path.exists(bdd_repr_path):
                    bdd_repr, gt_info = create_bdd_repr_from_ilp(instance_path, gt_info)
                    pickle.dump(bdd_repr, open(bdd_repr_path, "wb"))
                    pickle.dump(gt_info, open(sol_path, "wb"))
                if self.read_dual_converged:
                    if not os.path.exists(bdd_repr_conv_path):
                        bdd_repr = pickle.load(open(bdd_repr_path, 'rb'))
                        bdd_repr = solve_dual_bdd(bdd_repr, 1e-6, 20000, 0.5)
                        pickle.dump(bdd_repr, open(bdd_repr_conv_path, "wb"))
                    bdd_repr_path = bdd_repr_conv_path # Read dual converged instead.
                self.file_list.append({'instance_path': instance_path, 
                                    'bdd_repr_path': bdd_repr_path,
                                    'sol_path': sol_path, 
                                    'lp_size': os.path.getsize(instance_path)})
        def get_size(elem):
            return elem['lp_size']
        # Sort by size so that largest instances automatically go to end indices and thus to test set. 
        self.file_list.sort(key = get_size)

    def len(self):
        return len(self.file_list)

    def get(self, index):
        lp_path = self.file_list[index]['instance_path']
        bdd_repr_path = self.file_list[index]['bdd_repr_path']
        sol_path = self.file_list[index]['sol_path']
        gt_info = pickle.load(open(sol_path, 'rb'))
        bdd_repr = pickle.load(open(bdd_repr_path, 'rb'))
        return create_graph_from_bdd_repr(bdd_repr, gt_info, lp_path)