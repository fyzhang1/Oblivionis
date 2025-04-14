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


# train从这里获得data
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


        # 对forget部分data进行unlearn train
        unlearn_dataset = ForgetRetainDataset(**unlearn_splits, anchor=anchor)

        

        data["train"] = unlearn_dataset
        for split in unlearn_splits:
            data.pop(split)
    return data

# def get_data(data_cfg: DictConfig, mode="train", **kwargs):
#     data = {}
#     data_cfg = dict(data_cfg)
#     anchor = data_cfg.pop("anchor", "forget")
#     for split, dataset_cfgs in data_cfg.items():
#         data[split] = get_datasets(dataset_cfgs, **kwargs)
    
#     if mode == "train":
#         return data
#     elif mode == "unlearn":
#         unlearn_splits = {k: v for k, v in data.items() if k not in ("eval", "test")}

#         # Create unlearn and retain datasets
#         unlearn_dataset = ForgetRetainDataset(**unlearn_splits, anchor=anchor)
#         retain_dataset = ForgetRetainDataset(**unlearn_splits, anchor="retain")
#         print(f"Unlearn dataset data structure: {unlearn_dataset}")
#         print(f"Retain dataset data structure: {retain_dataset}")

#         # Assign datasets to data
#         data["forget"] = unlearn_dataset
#         data["retain"] = retain_dataset
        
#         # Remove only the original splits, preserving "forget" and "retain"
#         for split in list(unlearn_splits.keys()):
#             if split not in ("forget", "retain"):  # Protect "forget" and "retain"
#                 data.pop(split, None)  # Use None to avoid KeyError if key doesn't exist

#         print(f"11111 data structure: {data.keys()}")
#     return data


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
#     """
#     将数据分割并分配给各个客户端
    
#     Args:
#         data: 数据字典，包含train键，train是ForgetRetainDataset或包含forget/retain子键的字典
#         num_clients: 客户端数量，默认为3
#         target_client_idx: 接收forget数据的目标客户端索引，默认为0
    
#     Returns:
#         federated_data: 分配给各个客户端的数据字典
#     """
#     # 检查数据结构
#     # train_data = data.get("forget")
#     # if train_data is None:
#     #     raise ValueError("Train data not found")
    
#     # # 如果train_data是ForgetRetainDataset，直接访问其forget和retain属性
#     # if hasattr(train_data, "forget") and hasattr(train_data, "retain"):
#     #     forget_data = train_data.forget
#     #     retain_data = train_data.retain
#     # # 如果train_data是字典，检查是否包含forget和retain键
#     # elif isinstance(train_data, dict) and "forget" in train_data and "retain" in train_data:
#     #     forget_data = train_data["forget"]
#     #     retain_data = train_data["retain"]
#     # else:
#     #     raise ValueError("Failed to locate forget and retain data in the provided dataset")
    
#     # if forget_data is None or retain_data is None:
#     #     raise ValueError("Both forget and retain data must be provided")
    

#     forget_data = data["forget"]
#     retain_data = data["retain"]
#     # 创建联邦数据字典
#     federated_data = {}
    
#     # 为每个客户端创建空数据字典
#     for client_idx in range(num_clients):
#         federated_data[client_idx] = {}
    
#     # 目标客户端获取所有forget数据
#     federated_data[target_client_idx]["forget"] = forget_data
    
#     # 将retain数据平均分配给所有客户端
#     if isinstance(retain_data, dict):
#         # 如果retain_data是字典，需要对每个数据集进行分割
#         for dataset_name, dataset in retain_data.items():
#             data_len = len(dataset)
#             chunk_size = data_len // num_clients
#             for client_idx in range(num_clients):
#                 start_idx = client_idx * chunk_size
#                 end_idx = start_idx + chunk_size if client_idx < num_clients - 1 else data_len
#                 if dataset_name not in federated_data[client_idx]:
#                     federated_data[client_idx][dataset_name] = []
#                 federated_data[client_idx][dataset_name] = dataset[start_idx:end_idx]
#     else:
#         # 如果retain_data是单个数据集，直接分割
#         data_len = len(retain_data)
#         chunk_size = data_len // num_clients
#         for client_idx in range(num_clients):
#             start_idx = client_idx * chunk_size
#             end_idx = start_idx + chunk_size if client_idx < num_clients - 1 else data_len
#             federated_data[client_idx]["retain"] = retain_data[start_idx:end_idx]
    
#     # 为了适配ForgetRetainDataset，对每个客户端创建数据集
#     for client_idx in range(num_clients):
#         client_data = federated_data[client_idx]
        
#         # 只有目标客户端有forget数据，其他客户端的forget数据设为None
#         if client_idx != target_client_idx:
#             client_data["forget"] = None
        
#         # 使用ForgetRetainDataset包装数据
#         client_data = ForgetRetainDataset(
#             forget=client_data.get("forget"),
#             retain=client_data.get("retain"),
#             anchor="retain" if client_idx != target_client_idx else "forget"
#         )
        
#         federated_data[client_idx] = client_data
    
#     return federated_data


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
#         client_retain_data = retain_data[start_idx:end_idx]
        
#         if client_idx == target_client_idx:
#             federated_data[client_idx] = {
#                 "forget": forget_data,
#                 "retain": client_retain_data
#             }
#         else:
#             federated_data[client_idx] = {
#                 "retain": client_retain_data
#             }
    
#     return federated_data




def get_federated_data(data, num_clients=3, target_client_idx=0):
    """将数据集分配给多个客户端，目标客户端包含 forget 数据"""
    forget_data = data["forget"]
    retain_data = data["retain"]
    
    federated_data = {}
    data_len = len(retain_data)
    chunk_size = data_len // num_clients

    # print("All forget data:")
    # for i in range(len(data["forget"])):
    #     print(f"Sample {i}:", data["forget"][i])
    
    for client_idx in range(num_clients):
        start_idx = client_idx * chunk_size
        end_idx = start_idx + chunk_size if client_idx < num_clients - 1 else data_len
        # 使用 Subset 替代直接切片
        client_retain_data = Subset(retain_data, range(start_idx, end_idx))
        
        if client_idx == target_client_idx:
            federated_data[client_idx] = {
                "forget": forget_data,
                "retain": client_retain_data
            }
        else:
            federated_data[client_idx] = {
                "retain": client_retain_data
            }
    
    return federated_data

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
