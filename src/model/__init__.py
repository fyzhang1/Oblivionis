from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig, open_dict
import os
import torch
import logging
from typing import Union, Tuple
from vllm import LLM, SamplingParams

hf_home = os.getenv("HF_HOME", default=None)


logger = logging.getLogger(__name__)


def get_dtype(model_args):
    with open_dict(model_args):
        torch_dtype = model_args.pop("torch_dtype", None)
    if model_args["attn_implementation"] == "flash_attention_2":
        # This check handles https://github.com/Dao-AILab/flash-attention/blob/7153673c1a3c7753c38e4c10ef2c98a02be5f778/flash_attn/flash_attn_triton.py#L820
        # If you want to run at other precisions consider running "training or inference using
        # Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):`
        # decorator" or using an attn_implementation compatible with the precision in the model
        # config.
        assert torch_dtype in ["float16", "bfloat16"], ValueError(
            f"Invalid torch_dtype '{torch_dtype}' for the requested attention "
            f"implementation: 'flash_attention_2'. Supported types are 'float16' "
            f"and 'bfloat16'."
        )
    if torch_dtype == "float16":
        return torch.float16
    elif torch_dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def get_model(model_cfg: DictConfig) -> Union[Tuple[AutoModelForCausalLM, AutoTokenizer], Tuple[LLM, AutoTokenizer]]:
    assert model_cfg is not None and model_cfg.model_args is not None, ValueError(
        "Model config not found or model_args absent in configs/model."
    )
    model_args = model_cfg.model_args
    tokenizer_args = model_cfg.tokenizer_args
    
    # Check if we should use vLLM
    use_vllm = model_args.get("use_vllm", False)
    
    if use_vllm:
        return get_vllm_model(model_args, tokenizer_args)
    else:
        return get_transformers_model(model_args, tokenizer_args)


def get_vllm_model(model_args: DictConfig, tokenizer_args: DictConfig) -> Tuple[LLM, AutoTokenizer]:
    """Load model using vLLM backend."""
    try:
        # Convert DictConfig to dict for vLLM
        vllm_args = {
            "model": model_args.pretrained_model_name_or_path,
            "tensor_parallel_size": model_args.get("tensor_parallel_size", 1),
            "gpu_memory_utilization": model_args.get("gpu_memory_utilization", 0.9),
            "trust_remote_code": model_args.get("trust_remote_code", True),
            "max_model_len": model_args.get("max_model_len", 4096),
            "quantization": model_args.get("quantization", None),
            "enforce_eager": model_args.get("enforce_eager", False),
            "max_num_batched_tokens": model_args.get("max_num_batched_tokens", 4096),
            "max_num_seqs": model_args.get("max_num_seqs", 256),
        }
        model = LLM(**vllm_args)
    except Exception as e:
        logger.warning(
            f"Model {model_args.pretrained_model_name_or_path} requested with {model_args}"
        )
        raise ValueError(
            f"Error {e} while fetching model using vLLM.LLM()."
        )
    tokenizer = get_tokenizer(tokenizer_args)
    return model, tokenizer


def get_transformers_model(model_args: DictConfig, tokenizer_args: DictConfig) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model using Transformers backend."""
    torch_dtype = get_dtype(model_args)
    try:
        # Convert DictConfig to dict for transformers
        transformers_args = {
            "torch_dtype": torch_dtype,
            "pretrained_model_name_or_path": model_args.pretrained_model_name_or_path,
            "trust_remote_code": model_args.get("trust_remote_code", True),
            "cache_dir": hf_home
        }
        model = AutoModelForCausalLM.from_pretrained(**transformers_args)
    except Exception as e:
        logger.warning(
            f"Model {model_args.pretrained_model_name_or_path} requested with {model_args}"
        )
        raise ValueError(
            f"Error {e} while fetching model using AutoModelForCausalLM.from_pretrained()."
        )
    tokenizer = get_tokenizer(tokenizer_args)
    return model, tokenizer


def _add_or_replace_eos_token(tokenizer, eos_token: str) -> None:
    is_added = tokenizer.eos_token_id is None
    num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        logger.info("Add eos token: {}".format(tokenizer.eos_token))
    else:
        logger.info("Replace eos token: {}".format(tokenizer.eos_token))

    if num_added_tokens > 0:
        logger.info("New tokens have been added, make sure `resize_vocab` is True.")


def get_tokenizer(tokenizer_cfg: DictConfig):
    try:
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_cfg, cache_dir=hf_home)
    except Exception as e:
        error_message = (
            f"{'--' * 40}\n"
            f"Error {e} fetching tokenizer using AutoTokenizer.\n"
            f"Tokenizer requested from path: {tokenizer_cfg.get('pretrained_model_name_or_path', None)}\n"
            f"Full tokenizer config: {tokenizer_cfg}\n"
            f"{'--' * 40}"
        )
        raise RuntimeError(error_message)

    if tokenizer.eos_token_id is None:
        logger.info("replacing eos_token with <|endoftext|>")
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Setting pad_token as eos token: {}".format(tokenizer.pad_token))

    return tokenizer
