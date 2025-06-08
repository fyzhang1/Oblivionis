import torch
from torch.utils.data import Dataset


class ForgetRetainDataset(Dataset):
    # https://github.com/OPTML-Group/SOUL/blob/main/src/dataset/Base.py
    def __init__(self, forget, retain, anchor="forget"):
        """Wraps the forget retain dataset into unlearning dataset.

        Args:
            forget (Dataset): Forget Dataset
            retain (Dataset): Retain Dataset
            anchor (str, optional): Specifies which dataset to anchor while randomly sampling from the other dataset. Defaults to 'forget'.
        """
        self.forget = forget
        self.retain = retain
        self.anchor = anchor

    def __len__(self):
        """Ensures the sampled dataset matches the anchor dataset's length."""
        if self.anchor == "forget":
            assert self.forget is not None, ValueError(
                "forget dataset can't be None when anchor=forget"
            )
            return len(self.forget)
        elif self.anchor == "retain":
            assert self.retain is not None, ValueError(
                "retain dataset can't be None when anchor=retain"
            )
            return len(self.retain)
        else:
            raise NotImplementedError(f"{self.anchor} can be only forget or retain")

    def __getitem__(self, idx):
        """Returns the item from the anchor dataset.
        
        Returns:
            dict: A dictionary containing the input_ids, attention_mask, and labels.
        """
        if self.anchor == "forget":
            return self.forget[idx]
        elif self.anchor == "retain":
            return self.retain[idx]
        else:
            raise NotImplementedError(f"{self.anchor} can be only forget or retain")

def get_federated_data(forget_dataset, retain_dataset, num_clients, target_client_idx):
    """
    Split the datasets among clients in a federated learning setup.
    
    Args:
        forget_dataset (Dataset): Dataset containing samples to be forgotten
        retain_dataset (Dataset): Dataset containing samples to be retained
        num_clients (int): Number of clients in federated learning
        target_client_idx (int): Index of the client that will receive forget data
        
    Returns:
        dict: Dictionary mapping client indices to their ForgetRetainDataset instances
    """
    if target_client_idx >= num_clients:
        raise ValueError(f"Target client index {target_client_idx} must be less than number of clients {num_clients}")
    
    # Create an empty dictionary to store client datasets
    client_datasets = {}
    
    # Split retain dataset among all clients
    if retain_dataset is not None:
        dataset_size = len(retain_dataset)
        indices = torch.randperm(dataset_size).tolist()
        
        # Calculate the size of each client's portion
        client_size = dataset_size // num_clients
        
        for client_idx in range(num_clients):
            # Calculate start and end indices for this client's data
            start_idx = client_idx * client_size
            # For the last client, include any remaining samples
            end_idx = start_idx + client_size if client_idx < num_clients - 1 else dataset_size
            
            # Get indices for this client
            client_indices = indices[start_idx:end_idx]
            
            # Create subset of retain_dataset for this client
            from torch.utils.data import Subset
            client_retain = Subset(retain_dataset, client_indices)
            
            # Create ForgetRetainDataset for this client
            if client_idx == target_client_idx:
                # Target client gets both forget and retain data
                client_datasets[client_idx] = ForgetRetainDataset(
                    forget=forget_dataset,
                    retain=client_retain,
                    anchor="forget"
                )
            else:
                # Non-target clients only get retain data
                client_datasets[client_idx] = ForgetRetainDataset(
                    forget=None,
                    retain=client_retain,
                    anchor="retain"
                )
    
    return client_datasets
