# Modified from https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py

from typing import Dict, List, Optional, Union

import os
import logging
from transformers import Trainer
from torch.utils.data import Dataset
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from typing import Any
import torch
from transformers.utils import is_torch_available
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

logger = logging.getLogger(__name__)


class FinetuneTrainer(Trainer):
    def __init__(self, evaluator=None, template_args=None, *args, **kwargs):
        self.evaluator = evaluator
        self.template_args = template_args
        super().__init__(*args, **kwargs)

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        trial: Dict[str, Any] = None,
    ) -> Dict[str, float]:
        # Run a custom evaluator and save results
        if self.evaluator:
            if self.accelerator.is_local_main_process:
                eval_metrics = {}
                if self.accelerator.num_processes == 1:
                    run_dir = self._get_output_dir(trial=trial)
                    checkpoint_folder = (
                        f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                    )
                    output_dir = os.path.join(run_dir, checkpoint_folder, "evals")
                    os.makedirs(output_dir, exist_ok=True)
                    eval_args = {
                        "output_dir": output_dir,
                        "template_args": self.template_args,
                        "model": self.model,
                        "tokenizer": self.tokenizer,
                    }
                    eval_metrics = self.evaluator.evaluate(**eval_args)
                    eval_metrics = self.evaluator.summarize(eval_metrics)
                    self.log(eval_metrics)
                else:
                    logger.warning(
                        "Custom evaluator can be run with this Trainer only on a single GPU"
                    )
                return eval_metrics

        if eval_dataset is None:
            return {}
        # Run the default HF Trainer evaluate method when eval dataset is provided
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)


class VLLMModelWrapper(PreTrainedModel):
    def __init__(self, vllm_model):
        # Create a dummy config if none exists
        if not hasattr(vllm_model, 'config'):
            config = PretrainedConfig()
            config.model_type = "vllm"
            config.vocab_size = 32000  # Default value, adjust if needed
            config.hidden_size = 4096  # Default value, adjust if needed
            config.num_hidden_layers = 32  # Default value, adjust if needed
            config.num_attention_heads = 32  # Default value, adjust if needed
            config.intermediate_size = 11008  # Default value, adjust if needed
            config.hidden_act = "silu"  # Default value, adjust if needed
            config.max_position_embeddings = 4096  # Default value, adjust if needed
            config.initializer_range = 0.02  # Default value, adjust if needed
            config.rms_norm_eps = 1e-6  # Default value, adjust if needed
            config.use_cache = True  # Default value, adjust if needed
            config.pad_token_id = None  # Default value, adjust if needed
            config.bos_token_id = 1  # Default value, adjust if needed
            config.eos_token_id = 2  # Default value, adjust if needed
            config.tie_word_embeddings = False  # Default value, adjust if needed
            config.architectures = ["VLLMModel"]  # Default value, adjust if needed
        else:
            config = vllm_model.config
            
        super().__init__(config)
        self.vllm_model = vllm_model
        self.config = config
    
    def forward(self, *args, **kwargs):
        return self.vllm_model.generate(*args, **kwargs)
    
    def to(self, device):
        return self
    
    def __getattr__(self, name):
        return getattr(self.vllm_model, name)


class BaseTrainer(FinetuneTrainer):
    def __init__(self, *args, **kwargs):
        self.is_vllm = False
        if 'model' in kwargs and hasattr(kwargs['model'], '__class__'):
            self.is_vllm = kwargs['model'].__class__.__name__ == 'LLM'
        
        if self.is_vllm:
            # For vLLM models, we need to modify some Trainer attributes
            if 'args' in kwargs:
                kwargs['args'].label_names = ['labels']  # Set default label names
                kwargs['args'].remove_unused_columns = False  # Don't remove unused columns
                
            # Wrap the vLLM model
            kwargs['model'] = VLLMModelWrapper(kwargs['model'])
            
        super().__init__(*args, **kwargs)
        
    def _move_model_to_device(self, model, device):
        """Override the method to handle vLLM models."""
        if hasattr(model, 'to'):
            # For regular Transformers models
            return model.to(device)
        else:
            # For vLLM models, they are already on the correct device
            return model
            
    def compute_loss(self, model, inputs, return_outputs=False):
        """Override the method to handle vLLM models."""
        if not self.is_vllm:
            return super().compute_loss(model, inputs, return_outputs)
            
        # For vLLM models, we need to implement custom loss computation
        # This is a placeholder - you'll need to implement the actual loss computation
        # based on your specific requirements
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            labels = inputs.get("labels")
            
            if input_ids is not None and labels is not None:
                # Create sampling parameters
                from vllm import SamplingParams
                sampling_params = SamplingParams(
                    temperature=0.0,  # Use 0 temperature for deterministic output
                    max_tokens=labels.shape[1],
                    stop=None
                )
                
                # Generate outputs using vLLM
                outputs = model.generate(input_ids, sampling_params)
                
                # Compute loss (this is a placeholder - implement your actual loss computation)
                # You might need to convert the outputs to the format expected by your loss function
                loss = torch.tensor(0.0, device=input_ids.device)  # Placeholder
                
                if return_outputs:
                    return loss, outputs
                return loss
                
        raise NotImplementedError("Loss computation for vLLM models needs to be implemented for your specific use case")
        
    def _prepare_inputs(self, inputs):
        """Override to handle vLLM model inputs."""
        if not self.is_vllm:
            return super()._prepare_inputs(inputs)
            
        # For vLLM models, we need to handle inputs differently
        if isinstance(inputs, dict):
            # Move tensors to the correct device
            return {k: v.to(self.args.device) if torch.is_tensor(v) else v 
                   for k, v in inputs.items()}
        return inputs
