import ecole, os
import torch_geometric
import pickle
from tqdm import tqdm
from data.ilp_converters import create_bdd_repr_from_ilp, create_graph_from_bdd_repr
from data.gt_generator import presolve_and_generate_gt

def CreateSetCoverGenerator(n_rows, n_cols, density, max_coeff):
    return ecole.instance.SetCoverGenerator(n_rows, n_cols, density, max_coeff)

def CreateIndependentSetGenerator(n_nodes, edge_probability, affinity):
    return ecole.instance.IndependentSetGenerator(n_nodes = n_nodes, edge_probability = edge_probability, affinity = affinity)

def CreateCombinatorialAuctionGenerator():
    return ecole.instance.CombinatorialAuctionGenerator()

def CreateCapacitatedFacilityLocationGenerator():
    return ecole.instance.CapacitatedFacilityLocationGenerator()

class ILPRandomDiskDataset(torch_geometric.data.InMemoryDataset): #TODOAA: InMemoryDataset?
    def __init__(self, root, problem_string, generator_params, seed, num_samples):
        super().__init__(root=None, transform=None, pre_transform=None)
        problem_type, is_random = problem_string.split('_')
        assert is_random == 'Random' 
        
        if problem_type == 'SetCover':
            self.random_instance_generator = CreateSetCoverGenerator(**generator_params)
        elif problem_type == 'IndependentSet':
            self.random_instance_generator = CreateIndependentSetGenerator(**generator_params)
        elif problem_type == 'CombinatorialAuction':
            self.random_instance_generator = CreateCombinatorialAuctionGenerator(**generator_params)
        elif problem_type == 'CapacitatedFacilityLocation':
            self.random_instance_generator = CreateCapacitatedFacilityLocationGenerator(**generator_params)
        else:
            raise ValueError(f'Undefined problem_type: {problem_type}.')

        self.problem_type = problem_type
        self.num_samples = num_samples
        self.seed = seed
        self.random_instance_generator.seed(seed)
        self.root = os.path.join(root, self.problem_type + ''.join(['_' + str(k) + '_' + str(v) for k, v in generator_params.items()]) + '_seed_' + str(seed))
        os.makedirs(self.root, exist_ok=True)
        self.process_custom()

    @classmethod
    def from_config(cls, cfg, problem_string):
        generator_params = cfg.DATA[problem_string + '_PARAMS']
        num_samples = generator_params['num_samples']
        generator_params.pop('num_samples', None)
        return cls(
            root = cfg.DATA.RANDOM_DATA_ROOT,
            problem_string = problem_string,
            generator_params = generator_params,
            seed = cfg.SEED,
            num_samples = num_samples
        )

    def get_file_paths_custom(self, idx):
        return (os.path.join(self.root, f'ilp_instance_{idx}.mps'), # gurobi does not pick the constant at the end of objective in .lp format.
                os.path.join(self.root, f'ilp_bdd_{idx}.pkl'),
                os.path.join(self.root, f'ilp_solution_{idx}.pkl'))

    def process_custom(self):
        for i in tqdm(range(self.num_samples), f'Solving ILPs and saving at: {self.root}'):
            lp_path, bdd_path, sol_path = self.get_file_paths_custom(i)
            if os.path.exists(lp_path) and os.path.exists(sol_path) and os.path.exists(bdd_path):
                continue

            lp_stats = None
            ilp_stats = None
            while lp_stats is None:
                ilp_ecole = next(self.random_instance_generator)
                ilp_scip = ilp_ecole.as_pyscipopt()
                ilp_scip.writeProblem(lp_path)
                lp_stats, ilp_stats = presolve_and_generate_gt(lp_path)

            # Create BDD and save:
            gt_info = {"lp_stats": lp_stats, "ilp_stats": ilp_stats}
            bdd_repr, gt_info = create_bdd_repr_from_ilp(lp_path, gt_info)
            pickle.dump(bdd_repr, open(bdd_path, "wb"))
            pickle.dump(gt_info, open(sol_path, "wb"))

    def len(self):
        return self.num_samples

    def get(self, index):
        lp_path, bdd_path, sol_path = self.get_file_paths_custom(index)
        gt_info = pickle.load(open(sol_path, 'rb'))
        bdd_repr = pickle.load(open(bdd_path, 'rb'))
        return create_graph_from_bdd_repr(bdd_repr, gt_info, lp_path)
# ilp_scip.optimize()
# sol = ilp_scip.getBestSol()
# obj = ilp_scip.getSolObjVal(sol)
# if ilp_scip.getGap() > 0:
#     print(f'Gap of {ilp_scip.getGap()} found on {lp_path}')
# import gurobipy as gp
# from gurobipy import GRB
# ilp_gurobi = gp.read(lp_path)
# ilp_gurobi_presolved = ilp_gurobi.presolve()
# ilp_gurobi_presolved.write('test.lp')
# # ilp_ecole.write_problem(lp_path)
# best_sol = {}
# for var in variables:
#     best_sol.update({str(var): sol[var]})
# solution = { "obj": obj, "best_sol": best_sol }
