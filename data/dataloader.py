import torch 
from data.random_dataloader import ILPRandomDiskDataset
from data.disk_dataloader import ILPDiskDataset
from torch_geometric.loader import DataLoader

def get_ilp_gnn_loaders(cfg):
    all_train_datasets = []
    test_loaders = []
    test_datanames = []
    for data_name, test_fraction in zip(cfg.DATA.DATASETS, cfg.DATA.TEST_FRACTION):
        if '_Random' in data_name:
            full_dataset = ILPRandomDiskDataset.from_config(cfg, data_name)
        else:
            full_dataset = ILPDiskDataset.from_config(cfg, data_name)

        test_size = int(test_fraction * len(full_dataset))
        train_size = len(full_dataset) - test_size
        if test_size > 0 and train_size > 0:
            train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
        elif train_size > 0:
            train_dataset = full_dataset
            test_dataset = None 
        else:
            train_dataset = None 
            test_dataset = full_dataset
        
        if train_dataset is not None:
            all_train_datasets.append(train_dataset)
        if test_dataset is not None:
            # test datasets are not combined, they are kept separate to compute per dataset eval metrics.
            test_loaders.append(DataLoader(test_dataset, 
                                    batch_size=cfg.TEST.BATCH_SIZE, 
                                    shuffle=False, 
                                    follow_batch = ['objective', 'rhs_vector', 'edge_index_var_con'], 
                                    num_workers = cfg.DATA.NUM_WORKERS))
            test_datanames.append(data_name)
    combined_train_loader = None
    if len(all_train_datasets) > 0:
        combined_train_dataset = torch.utils.data.ConcatDataset(all_train_datasets)
        combined_train_loader = DataLoader(combined_train_dataset, 
                                        batch_size=cfg.TRAIN.BATCH_SIZE, 
                                        shuffle=True, 
                                        follow_batch = ['objective', 'rhs_vector', 'edge_index_var_con'], 
                                        num_workers = cfg.DATA.NUM_WORKERS)

    return combined_train_loader, test_loaders, test_datanames