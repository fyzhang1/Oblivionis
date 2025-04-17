from typing import Dict, Any, Union
from omegaconf import DictConfig
from torch.utils.data import Subset

from data.qa import (
    QADataset,
    QAwithIdkDataset,
)
from data.collators import (
    DataCollatorForSupervisedDataset,
)
from data.unlearn import ForgetRetainDataset
from data.pretraining import PretrainingDataset, CompletionDataset

DATASET_REGISTRY: Dict[str, Any] = {}
COLLATOR_REGISTRY: Dict[str, Any] = {}


def _register_data(data_class):
    DATASET_REGISTRY[data_class.__name__] = data_class


def _register_collator(collator_class):
    COLLATOR_REGISTRY[collator_class.__name__] = collator_class


def _load_single_dataset(dataset_name, dataset_cfg: DictConfig, **kwargs):
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
    dataset = {}
    for dataset_name, dataset_cfg in dataset_cfgs.items():
        access_name = dataset_cfg.get("access_key", dataset_name)
        dataset[access_name] = _load_single_dataset(dataset_name, dataset_cfg, **kwargs)
    if len(dataset) == 1:
        # return a single dataset
        return list(dataset.values())[0]
    # return mapping to multiple datasets
    return dataset


def get_data(data_cfg: DictConfig, mode="train", **kwargs):
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


# def get_federated_data(data, num_clients=3, target_client_idx=0):
#     """将数据集分配给多个客户端，目标客户端包含 forget 数据"""
#     forget_data = data["forget"]
#     retain_data = data["retain"]
    
#     federated_data = {}
#     data_len = len(retain_data)
#     chunk_size = data_len // num_clients
    
#     for client_idx in range(num_clients):
#         start_idx = client_idx * chunk_size
#         end_idx = start_idx + chunk_size if client_idx < num_clients - 1 else data_len
        
#         # 直接使用列表切片而不是Subset
#         # 提取实际数据项到列表中
#         client_retain_items = []
#         for i in range(start_idx, end_idx):
#             client_retain_items.append(retain_data[i])
        
#         if client_idx == target_client_idx:
#             federated_data[client_idx] = {
#                 "forget": forget_data,
#                 "retain": client_retain_items
#             }
#         else:
#             federated_data[client_idx] = {
#                 "retain": client_retain_items
#             }
    
    
#     return federated_data

def get_federated_data(data, num_clients=3, target_client_idx=0, anchor="retain"):
    """
    联邦数据分配（完全类型一致版）
    确保所有客户端的 retain 和 forget 均为 ForgetRetainDataset 类型
    """
    # 提取原始数据集（必须均为 ForgetRetainDataset 类型）
    original_retain: ForgetRetainDataset = data["retain"]
    original_forget: ForgetRetainDataset = data["forget"]

    # 参数校验
    if target_client_idx >= num_clients:
        raise ValueError(f"Target client index {target_client_idx} >= num_clients {num_clients}")
    if not isinstance(original_retain, ForgetRetainDataset) or not isinstance(original_forget, ForgetRetainDataset):
        raise TypeError("输入数据集必须为 ForgetRetainDataset 类型")

    # 分割 retain 数据集
    retain_len = len(original_retain.retain)  # 原始 retain 数据的实际长度
    chunk_size = retain_len // num_clients

    client_datasets = {}
    for client_idx in range(num_clients):
        # 分割 retain 的索引（基于原始 retain 数据的真实长度）
        start = client_idx * chunk_size
        end = start + chunk_size if client_idx < num_clients-1 else retain_len
        retain_indices = list(range(start, end))

        # 创建客户端的 retain 数据集（保持 ForgetRetainDataset 类型）
        client_retain = ForgetRetainDataset(
            forget=None,  # retain 子集不含 forget 数据
            retain=Subset(original_retain.retain, retain_indices),  # 分割原始 retain 数据
            anchor="retain"  # 强制设置为 retain 锚点
        )

        # 分配 forget 数据
        if client_idx == target_client_idx:
            # 目标客户端获得完整 forget 数据集
            client_forget = original_forget
        else:
            # 非目标客户端的 forget 设为空
            client_forget = ForgetRetainDataset(forget=None, retain=None, anchor="retain")

        # 创建客户端最终数据集
        client_dataset = ForgetRetainDataset(
            forget=client_forget,
            retain=client_retain,
            anchor=anchor
        )

        client_datasets[client_idx] = client_dataset

        # 类型验证
        print(f"客户端 {client_idx} 数据类型验证:")
        print(f"  forget 类型: {type(client_dataset.forget)}")  # 应输出 ForgetRetainDataset
        print(f"  retain 类型: {type(client_dataset.retain)}")  # 应输出 ForgetRetainDataset

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
