import torch
import torch_geometric
import numpy as np
import torch.utils.data
import BDD.ILP_instance_py as ilp_instance_bbd
import BDD.bdd_cuda_learned_mma_py as bdd_solver
import pickle
import gurobipy as gp

class IndexManager():
    def __init__(self, feature_names):
        self.name_to_index = {}
        self.index_to_name = {}
        for idx, name in enumerate(feature_names):
            self.name_to_index[name] = idx
            self.index_to_name[idx] = name

    def get_index(self, name):
        col_idx = self.name_to_index[name]
        return col_idx

    def feature_used(self, name):
        return name in self.name_to_index
    
    @property
    def num_features(self):
        return len(self.name_to_index)

def create_normalized_bdd_instance(ilp_path):
    ilp_gurobi = gp.read(ilp_path)
    ilp_gurobi.setAttr('ObjCon', 0.0) # Remove constant from objective and re-write LP.
    ilp_gurobi.update()
    ilp_gurobi.write(ilp_path)
    obj_multiplier = 1.0
    if ilp_gurobi.ModelSense == -1: # Maximization
        obj_multiplier = -1.0
    variables = ilp_gurobi.getVars()
    cons = ilp_gurobi.getConstrs()
    ilp_bdd = ilp_instance_bbd.ILP_instance()
    objs = []
    var_names = []
    for var in variables:
        assert var.VType == 'B', f'Variable {var} is not binary in file {ilp_path} and instead of type {var.vtype()}'
        objs.append(var.Obj)
        var_names.append(var.VarName)

    # Scale the objective vector to [-1, 1]. (This will also change the gt_obj later).
    obj_multiplier = obj_multiplier / np.abs(np.array(objs)).max()
    for i in range(len(objs)):
        ilp_bdd.add_new_variable_with_obj(var_names[i], obj_multiplier * float(objs[i]))

    max_coeff = 0
    for con in cons:
        multiplier = 1.0
        rhs_value = con.RHS
        if con.Sense == '=':
            ineq_type = ilp_instance_bbd.equal
        elif con.Sense == '<':
            ineq_type = ilp_instance_bbd.smaller_equal
        else:
            assert(con.Sense == '>')
            # Converting all constraints to <= or = representation for GNN.
            multiplier = -1.0
            ineq_type = ilp_instance_bbd.smaller_equal
        
        constraint_var_names = []; constraint_var_coefficients =  []
        constraint_exp = ilp_gurobi.getRow(con)
        for i in range(constraint_exp.size()):
            var = constraint_exp.getVar(i).VarName
            coeff = constraint_exp.getCoeff(i)
            assert(int(coeff) == coeff)
            assert(int(rhs_value) == rhs_value)
            constraint_var_names.append(str(var))
            constraint_var_coefficients.append(int(coeff * multiplier))
            max_coeff = max(max_coeff, coeff)
        ilp_bdd.add_new_constraint(con.ConstrName, constraint_var_names, constraint_var_coefficients, int(rhs_value * multiplier), ineq_type)
    return ilp_bdd, obj_multiplier

def map_solution_to_index(bdd_ilp_instance, solution_dict):
    solution = np.zeros((bdd_ilp_instance.nr_variables() + 1), dtype = np.float32)
    for var_name, var_value in solution_dict.items():
        try:
            var_index = bdd_ilp_instance.get_var_index(var_name)
            solution[var_index] = float(var_value)
        except:
            breakpoint()
    return solution

def create_bdd_repr_from_ilp(ilp_path, gt_info):
    bdd_ilp_instance, obj_multiplier = create_normalized_bdd_instance(ilp_path)
    gt_info['obj_multiplier'] = obj_multiplier
    # Map variable names to indices for converting solution from dict to array:
    if 'lp_stats' in gt_info:
        gt_info['lp_stats']['sol'] = map_solution_to_index(bdd_ilp_instance, gt_info['lp_stats']['sol'])
    if 'ilp_stats' in gt_info:
        gt_info['ilp_stats']['sol'] = map_solution_to_index(bdd_ilp_instance, gt_info['ilp_stats']['sol'])
    solver = bdd_solver.bdd_cuda_learned_mma(bdd_ilp_instance)
    assert solver.nr_primal_variables() == bdd_ilp_instance.nr_variables(), f'Found {solver.nr_primal_variables()} variables in solver and {bdd_ilp_instance.nr_variables()} in ILP read by BDD solver.'
    assert solver.nr_bdds() == bdd_ilp_instance.nr_constraints(), f'Found {solver.nr_bdds()} BDDs in solver and {bdd_ilp_instance.nr_constraints()} constraints in ILP {ilp_path} read by BDD solver.'
    
    num_vars = solver.nr_primal_variables() + 1 # +1 due to terminal node.
    num_cons = solver.nr_bdds()
    num_layers = solver.nr_layers()
    var_indices = torch.empty((solver.nr_layers()), dtype = torch.int32, device = 'cuda')
    solver.primal_variable_index(var_indices.data_ptr())
    con_indices = torch.empty((solver.nr_layers()), dtype = torch.int32, device = 'cuda')
    solver.bdd_index(con_indices.data_ptr())
    var_indices = var_indices.cpu().numpy()
    con_indices = con_indices.cpu().numpy()
    objective = np.concatenate((bdd_ilp_instance.objective(), [0]))
    assert(objective.shape[0] == num_vars)

    # Encode constraints as features assuming that constraints are linear:
    try:
        coefficients = solver.constraint_matrix_coefficients(bdd_ilp_instance)
    except:
        coefficients = torch.ones((solver.nr_layers()), dtype = torch.float32, device = 'cuda').cpu().numpy()
        # breakpoint()
    bounds = bdd_ilp_instance.variable_constraint_bounds()
    # # Constraint features:
    # # bounds containst value of lb, ub meaning: lb <= constraint <= ub.
    bounds = torch.as_tensor(bounds)
    lb_cons = bounds[num_vars - 1:, 0]
    ub_cons = bounds[num_vars - 1:, 1]
    # lb <=(geq type) a^{T}x <=(leq type) ub. (lb can be equal to ub then the constraint is both leq and geq.)
    leq_cons = lb_cons <= np.iinfo(np.intc).min
    geq_cons = ub_cons >= np.iinfo(np.intc).max
    assert(~torch.any(torch.logical_and(leq_cons, geq_cons)))
    leq_type = torch.ones((num_cons))
    geq_type = torch.ones((num_cons))
    leq_type[geq_cons] = 0
    geq_type[leq_cons] = 0
    if torch.abs((1 - leq_type) * geq_type).max() > 0:
        raise ValueError('all constraints should be <= or = type')
    rhs_vector = lb_cons
    rhs_vector[leq_cons] = ub_cons[leq_cons]

    bdd_repr = {
                    "solver_data": pickle.dumps(solver, -1), # bytes representation of bdd cuda solver internal variables.
                    "num_vars": num_vars, "num_cons": num_cons, "num_layers": num_layers,
                    "var_indices": var_indices, "con_indices": con_indices,
                    "objective": objective, 
                    "coeffs": coefficients, "rhs_vector": rhs_vector.numpy(),
                    "constraint_type": leq_type.numpy() # Contains 1 for <= constraint and 0 for equality, where >= constraints should not be present.
                }
    return bdd_repr, gt_info

def solve_dual_bdd(bdd_repr, improvement_slope, num_iterations, omega):
    solver = pickle.loads(bdd_repr['solver_data'])
    solver.non_learned_iterations(omega, num_iterations, improvement_slope)
    solver.distribute_delta() # make deferred min-marginals zero.
    bdd_repr['solver_data'] = pickle.dumps(solver, -1) # Overwrite bdd representation with update costs.
    return bdd_repr

def create_graph_from_bdd_repr(bdd_repr, gt_info, file_path):
    graph = BipartiteVarConDataset(num_vars = bdd_repr['num_vars'], num_cons = bdd_repr['num_cons'], num_layers = bdd_repr['num_layers'],
                                var_indices = torch.from_numpy(bdd_repr['var_indices']).to(torch.long), 
                                con_indices = torch.from_numpy(bdd_repr['con_indices']).to(torch.long), 
                                objective = torch.from_numpy(bdd_repr['objective']).to(torch.float32),
                                con_coeffs = torch.from_numpy(bdd_repr['coeffs']).to(torch.float32), 
                                rhs_vector = torch.from_numpy(bdd_repr['rhs_vector']).to(torch.float32),
                                con_type = torch.from_numpy(bdd_repr['constraint_type']).to(torch.float32))

    graph.num_nodes = graph.num_vars + graph.num_cons
    if gt_info is not None: # If maximization problem was converted to minimization by *(-1) then gt_obj should change as well.
        if 'lp_stats' in gt_info:
            gt_info['lp_stats']['obj'] = gt_info['lp_stats']['obj'] * gt_info['obj_multiplier']
        if 'ilp_stats' in gt_info:
            gt_info['ilp_stats']['obj'] = gt_info['ilp_stats']['obj'] * gt_info['obj_multiplier']
    graph.gt_info = gt_info
    graph.solver_data = bdd_repr['solver_data']
    graph.file_path = file_path
    return graph

class BipartiteVarConDataset(torch_geometric.data.Data):
    def __init__(self, num_vars, num_cons, num_layers, var_indices, con_indices, objective, con_coeffs, rhs_vector, con_type):
        super(BipartiteVarConDataset, self).__init__()
        # super().__init__()
        self.num_vars = num_vars
        self.num_cons = num_cons
        self.num_layers = num_layers
        self.objective = objective
        self.con_coeff = con_coeffs
        self.rhs_vector = rhs_vector
        self.con_type = con_type 
        if var_indices is not None:
            assert(con_coeffs.shape == var_indices.shape)
            assert(con_coeffs.shape == con_indices.shape)
            assert(torch.numel(self.rhs_vector) == self.num_cons)
            self.edge_index_var_con = torch.stack((var_indices, con_indices))
            self.num_edges = torch.numel(var_indices)
        else:
            self.edge_index_var_con = None
            self.num_edges = None

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_var_con':
            return torch.tensor([[self.num_vars], [self.num_cons]])
        # if key == 'var_indices':
        #     return torch.tensor([self.num_vars])
        # if key == 'var_indices':
        #     return torch.tensor([self.num_vars])
        else:
            return super().__inc__(key, value, *args, **kwargs)

#create_normalized_bdd_instance('/home/ahabbas/data/learnDBCA/cv_structure_pred/cell-tracking-AISTATS-2020/large/flywing_100_1/instances/5.lp')
#create_normalized_bdd_instance('/BS/discrete_opt/work/datasets/graph_matching/hotel_house/hotel/energy_hotel_frame29frame36.lp')