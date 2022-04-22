import os

root_dir = '/home/ahabbas/data/learnDBCA/shape_matching'

for path, subdirs, files in os.walk(root_dir):
    for instance_name in sorted(files):
        if not instance_name.endswith('.lp'):
            continue
        if 'partial' in instance_name or 'noniso' in instance_name:
            continue
        
        new_instance_name = instance_name.replace('.lp', 'too_easy.lp')

        instance_path = os.path.join(path, instance_name)
        bdd_repr_path = instance_path.replace('.lp', '_bdd_repr.pkl')
        sol_name = instance_name.replace('.lp', '.pkl')
        sol_path = os.path.join(path.replace('instances', 'solutions'), sol_name)


        new_instance_path = os.path.join(path, new_instance_name)
        new_bdd_repr_path = new_instance_path.replace('.lp', '_bdd_repr.pkl')
        new_sol_name = new_instance_name.replace('.lp', '.pkl')
        new_sol_path = os.path.join(path.replace('instances', 'solutions'), new_sol_name)
        os.rename(instance_path, new_instance_path)
        os.rename(bdd_repr_path, new_bdd_repr_path)
        os.rename(sol_path, new_sol_path)

        print(f'renamed {instance_path} to {new_instance_path}')
