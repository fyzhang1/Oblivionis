from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig, open_dict
import os
import torch
import logging
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

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


def get_model(model_cfg: DictConfig):
    assert model_cfg is not None and model_cfg.model_args is not None, ValueError(
        "Model config not found or model_args absent in configs/model."
    )
    model_args = model_cfg.model_args
    tokenizer_args = model_cfg.tokenizer_args
    torch_dtype = get_dtype(model_args)

    # Check if we need to load a PEFT model
    use_lora = model_args.get("use_lora", False)
    lora_config = model_args.get("lora_config", None)
    lora_model_path = model_args.get("lora_model_path", None)

    try:
        # Convert DictConfig to dict for transformersAdd commentMore actions
        transformers_args = {
            "torch_dtype": torch_dtype,
            "pretrained_model_name_or_path": model_args.pretrained_model_name_or_path,
            "trust_remote_code": model_args.get("trust_remote_code", True),
            "cache_dir": hf_home
        }
        
        if lora_model_path:
            # Load existing PEFT model
            logger.info(f"Loading existing PEFT model from {lora_model_path}")
            model = PeftModel.from_pretrained(
                AutoModelForCausalLM.from_pretrained(**transformers_args),
                lora_model_path
            )
        elif use_lora:
            # Create new LoRA model
            logger.info("Creating new LoRA model")
            base_model = AutoModelForCausalLM.from_pretrained(**transformers_args)
            
            # Default LoRA configuration
            default_lora_config = {
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "lora_dropout": 0.1,
                "bias": "none",
                "task_type": TaskType.CAUSAL_LM,
            }
            
            # Update with user config if provided
            if lora_config:
                # Convert DictConfig to regular dict to avoid JSON serialization issues
                if hasattr(lora_config, '_content'):
                    # Handle DictConfig
                    lora_config_dict = {}
                    for key, value in lora_config.items():
                        if hasattr(value, '_content'):
                            # Convert ListConfig to regular list
                            lora_config_dict[key] = list(value)
                        else:
                            lora_config_dict[key] = value
                    default_lora_config.update(lora_config_dict)
                else:
                    # Handle regular dict
                    default_lora_config.update(lora_config)
            
            peft_config = LoraConfig(**default_lora_config)
            model = get_peft_model(base_model, peft_config)
            
            # Print trainable parameters info
            model.print_trainable_parameters()
        else:
            # Load standard model without LoRA
            model = AutoModelForCausalLM.from_pretrained(**transformers_args)
    
    except Exception as e:
        logger.warning(
            f"Model {model_args.pretrained_model_name_or_path} requested with {model_cfg.model_args}"
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
