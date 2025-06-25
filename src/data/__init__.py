# Update: Xinyu Yan, 2025-06-21
# TL;DR: This file is used to register the datasets and collators

from omegaconf import DictConfig
from typing import Dict, Any, Union
from torch.utils.data import Subset

from data.collators import DataCollatorForSupervisedDataset
from data.pretraining import PretrainingDataset, CompletionDataset
from data.qa import QADataset, QAwithIdkDataset
from data.unlearn import ForgetRetainDataset


DATASET_REGISTRY: Dict[str, Any] = {}
COLLATOR_REGISTRY: Dict[str, Any] = {}


def _register_data(data_class):
    """Register a dataset class to the global registry"""
    DATASET_REGISTRY[data_class.__name__] = data_class


def _register_collator(collator_class):
    """Register a collator class to the global registry"""
    COLLATOR_REGISTRY[collator_class.__name__] = collator_class


def _load_single_dataset(dataset_name, dataset_cfg: DictConfig, **kwargs):
    """Load a single dataset based on configuration"""
    dataset_handler_name = dataset_cfg.get("handler")
    assert dataset_handler_name is not None, ValueError(
        f"{dataset_name} handler not set"
    )
    dataset_handler = DATASET_REGISTRY.get(dataset_handler_name)
    if dataset_handler is None:
        raise NotImplementedError(
            f"{dataset_handler_name} not implemented or not registered"
        )
    dataset_args = dataset_cfg.args
    return dataset_handler(**dataset_args, **kwargs)


def get_datasets(dataset_cfgs: Union[Dict, DictConfig], **kwargs):
    """Load multiple datasets based on configuration"""
    dataset = {}
    for dataset_name, dataset_cfg in dataset_cfgs.items():
        access_name = dataset_cfg.get("access_key", dataset_name)
        dataset[access_name] = _load_single_dataset(dataset_name, dataset_cfg, **kwargs)
    if len(dataset) == 1:
        # Return a single dataset
        return list(dataset.values())[0]
    # Return mapping to multiple datasets
    return dataset


def get_data(data_cfg: DictConfig, mode="train", **kwargs):
    """Get data based on configuration and mode (train or unlearn)"""
    data = {}
    data_cfg = dict(data_cfg)
    anchor = data_cfg.pop("anchor", "forget")
    for split, dataset_cfgs in data_cfg.items():
        data[split] = get_datasets(dataset_cfgs, **kwargs)

    if mode == "train":
        return data
    elif mode == "unlearn":
        unlearn_splits = {k: v for k, v in data.items() if k not in ("eval", "test")}

        # Create unlearn and retain datasets
        unlearn_dataset = ForgetRetainDataset(**unlearn_splits, anchor=anchor)
        retain_dataset = ForgetRetainDataset(**unlearn_splits, anchor="retain")
        print(f"Unlearn dataset data structure: {unlearn_dataset}")
        print(f"Retain dataset data structure: {retain_dataset}")

        # Assign datasets to data
        data["forget"] = unlearn_dataset
        data["retain"] = retain_dataset

        # Remove only the original splits, preserving "forget" and "retain"
        for split in list(unlearn_splits.keys()):
            if split not in ("forget", "retain"):  # Protect "forget" and "retain"
                data.pop(split, None)  # Use None to avoid KeyError if key doesn't exist

        print(f"11111 data structure: {data.keys()}")
    return data


def _get_single_collator(collator_name: str, collator_cfg: DictConfig, **kwargs):
    """Load a single collator based on configuration"""
    collator_handler_name = collator_cfg.get("handler")
    assert collator_handler_name is not None, ValueError(
        f"{collator_name} handler not set"
    )
    collator_handler = COLLATOR_REGISTRY.get(collator_handler_name)
    if collator_handler is None:
        raise NotImplementedError(
            f"{collator_handler_name} not implemented or not registered"
        )
    collator_args = collator_cfg.args
    return collator_handler(**collator_args, **kwargs)


def get_collators(collator_cfgs, **kwargs):
    """Load multiple collators based on configuration"""
    collators = {}
    for collator_name, collator_cfg in collator_cfgs.items():
        collators[collator_name] = _get_single_collator(
            collator_name, collator_cfg, **kwargs
        )
    if len(collators) == 1:
        # return a single collator
        return list(collators.values())[0]
    # return collators in a dict
    return collators


def get_federated_data(data, num_clients=3, target_client_idx=0, anchor="retain"):
    """
    Distribute data for federated learning scenarios
    Ensure that all clients' retain and forget are ForgetRetainDataset types
    """
    # Extract original datasets (must be ForgetRetainDataset types)
    original_retain: ForgetRetainDataset = data["retain"]
    original_forget: ForgetRetainDataset = data["forget"]

    # Parameter validation
    if target_client_idx >= num_clients:
        raise ValueError(
            f"Target client index {target_client_idx} >= num_clients {num_clients}"
        )
    if not isinstance(original_retain, ForgetRetainDataset) or not isinstance(
        original_forget, ForgetRetainDataset
    ):
        raise TypeError("Input datasets must be ForgetRetainDataset types")

    # Split retain dataset
    retain_len = len(original_retain.retain)  # Actual length of original retain data
    chunk_size = retain_len // num_clients

    client_datasets = {}
    for client_idx in range(num_clients):
        # Split retain indices (based on actual length of original retain data)
        start = client_idx * chunk_size
        end = start + chunk_size if client_idx < num_clients - 1 else retain_len
        retain_indices = list(range(start, end))

        # Create client's retain dataset (keep ForgetRetainDataset type)
        client_retain = ForgetRetainDataset(
            forget=None,  # retain subset does not contain forget data
            retain=Subset(
                original_retain.retain, retain_indices
            ),  # Split original retain data
            anchor="retain",  # Force set to retain anchor
        )

        # Assign forget data
        if client_idx == target_client_idx:
            # Target client gets full forget dataset
            client_forget = original_forget
        else:
            # Non-target client's forget is set to None
            client_forget = ForgetRetainDataset(
                forget=None, retain=None, anchor="retain"
            )

        # Create client final dataset
        client_dataset = ForgetRetainDataset(
            forget=client_forget, retain=client_retain, anchor=anchor
        )

        client_datasets[client_idx] = client_dataset

        # Type validation
        print(f"Client {client_idx} Data Type Validation:")
        print(
            f"Forget Type: {type(client_dataset.forget)}"
        )  # Should output ForgetRetainDataset
        print(
            f"Retain Type: {type(client_dataset.retain)}"
        )  # Should output ForgetRetainDataset

    return client_datasets


# Register datasets
_register_data(QADataset)
_register_data(QAwithIdkDataset)
_register_data(PretrainingDataset)
_register_data(CompletionDataset)

# Register composite datasets used in unlearning
# groups: unlearn
_register_data(ForgetRetainDataset)

# Register collators
_register_collator(DataCollatorForSupervisedDataset)