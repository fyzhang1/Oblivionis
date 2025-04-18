import torch
import logging

logger = logging.getLogger(__name__)

def FedAvg(client_state_dicts, global_model_state_dict=None):
    """
    执行FedAvg策略的模型参数聚合
    
    Args:
        client_state_dicts: 客户端模型状态字典列表
        global_model_state_dict: 全局模型状态字典，用于处理空参数情况
        
    Returns:
        全局聚合后的状态字典
    """
    global_state_dict = {}
    for key in client_state_dicts[0].keys():
        if isinstance(client_state_dicts[0][key], torch.Tensor):
            valid_params = [state_dict[key] for state_dict in client_state_dicts if state_dict[key].numel() > 0]
            if valid_params:
                global_state_dict[key] = torch.stack(valid_params).mean(dim=0)
            else:
                if global_model_state_dict is not None:
                    global_state_dict[key] = global_model_state_dict[key]
                    logger.warning(f"All clients have empty parameter for {key}, using global model parameter")
                else:
                    raise ValueError(f"All clients have empty parameter for {key} and no global model provided")
        else:
            global_state_dict[key] = client_state_dicts[0][key]
    
    return global_state_dict