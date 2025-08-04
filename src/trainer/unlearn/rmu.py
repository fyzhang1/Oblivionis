"""Borrowed implementation from https://github.com/centerforaisafety/wmdp/blob/main/rmu/unlearn.py"""

import re
import torch
import deepspeed
from trainer.unlearn.grad_diff import GradDiff
from peft import PeftModel

class RMU(GradDiff):
    def __init__(
        self,
        module_regex=r"model\.layers\.7",
        trainable_params_regex=[r"model\.layers\.(5|6|7)\.mlp\.down_proj\.weight"],
        steering_coeff=20,
        *args,
        **kwargs,
    ):
        """
        RMU Trainer that fine-tunes only specific layers and parameters using regex-based filtering.

        Args:
            module_path (str): Regex pattern to match module names.
            trainable_param_paths (list of str): List of regex patterns for trainable parameters.
        """
        super().__init__(*args, **kwargs)

        # Create reference model if not already set
        if self.ref_model is None:
            print("Creating reference model...")
            self.ref_model = self._prepare_ref_model(self.model)
            # Move reference model to CPU to save GPU memory
            print("Moving reference model to CPU to save GPU memory...")
            self.ref_model = self.ref_model.cpu()

        # Unfreeze only the selected parameters
        self.trainable_params_regex = (
            trainable_params_regex  # Regex for selecting params
        )

        # Get actual module references
        self.module_regex = module_regex  # Regex for selecting modules
        self.model_module = self._get_matching_module(self.model, self.module_regex)
        self.ref_module = self._get_matching_module(self.ref_model, self.module_regex)
        self.steering_coeff = steering_coeff
        self.control_vec = None

    def create_optimizer(self):
        self._freeze_all_params(self.model, False)
        # This makes the optimizer to select only trainable params
        self._set_trainable_params(self.model, self.trainable_params_regex, True)
        super().create_optimizer()
        self._freeze_all_params(self.model, True)

    def _get_matching_module(self, model, module_regex):
        # print("Available modules:")
        # for name, module in model.named_modules():
        #     print(f"  {name}")
        # Check if the model is a LoRA model
        is_lora_model = isinstance(model, PeftModel) or any(".lora_" in name for name, _ in model.named_modules())
        print(f"Model is {'a LoRA model' if is_lora_model else 'not a LoRA model'}")

        # Use re.search to find matching modules
        for name, module in model.named_modules():
            name = name.replace(".default", "")
            if re.search(module_regex, name):
                print(f"Found matching module: {name}")
                return module

        # If no match is found, raise a detailed error
        raise ValueError(f"No module matched with {module_regex}. Available modules printed above.")

    # def _get_matching_module(self, model, module_regex):
    #     """Returns a single module matching the given regex from a DeepSpeed/DDP-wrapped model."""
    #     # Handle DeepSpeed and DDP-wrapped models by accessing the underlying module
    #     if isinstance(model, deepspeed.DeepSpeedEngine):
    #         model = model.module  # Extract the actual PyTorch model inside

    #     matched_modules = {
    #         name: module
    #         for name, module in model.named_modules()
    #         if re.fullmatch(module_regex, name)
    #     }



    #     if len(matched_modules) > 1:
    #         raise ValueError(
    #             f"More than one module matched with {module_regex}: {list(matched_modules.keys())}"
    #         )
    #     elif not matched_modules:
    #         raise ValueError(f"No module matched with {module_regex}")

    #     return next(iter(matched_modules.values()))  # Return the single matched module

    def _freeze_all_params(self, model, requires_grad=True):
        """Freeze all parameters in the model initially."""
        for param in model.parameters():
            param.requires_grad = requires_grad

    def _set_trainable_params(self, model, trainable_params_regex, requires_grad=True):
        """Unfreeze specific parameters that match the regex patterns."""
        for name, param in model.named_parameters():
            if any(re.fullmatch(pattern, name) for pattern in trainable_params_regex):
                param.requires_grad = requires_grad
                # print(f"{name}:requires_grad\t{requires_grad}")

    def forward_with_cache(self, model, inputs, module, no_grad=True):
        """Performs a forward pass while caching the output of a specified module."""
        cache = []

        def hook(module, input, output):
            if isinstance(output, tuple):
                cache.append(output[0])
            else:
                cache.append(output)
            return None

        hook_handle = module.register_forward_hook(hook)
        with torch.set_grad_enabled(not (no_grad)):
            outputs = model(**inputs)
        hook_handle.remove()
        return cache[0], outputs

    def get_control_vector(self, dim):
        if self.control_vec is None:
            random_vector = torch.rand(1, 1, dim)
            self.control_vec = (
                random_vector / torch.norm(random_vector) * self.steering_coeff
            )
        return self.control_vec

    def compute_activation_loss(self, activation1, activation2, mask):
        squared_diff = torch.nn.functional.mse_loss(
            activation1, activation2, reduction="none"
        )  # Shape (b, s, d)
        expanded_mask = mask.unsqueeze(-1).expand_as(squared_diff)  # Shape: [b, s, d]
        squared_diff_sum = (
            (squared_diff * expanded_mask).mean(dim=2).sum(dim=(1))
        )  # Shape: [b, 1]
        num_tokens = mask.sum(dim=-1, keepdim=True)  # Sum over seq_len, Shape: [b, 1]
        return (squared_diff_sum / num_tokens).mean()

    def compute_retain_loss(self, model, retain_inputs):
        retain_loss = 0.0

        if self.retain_loss_type == "EMBED_DIFF":
            model_retain_activations, _ = self.forward_with_cache(
                model, retain_inputs, module=self.model_module, no_grad=False
            )

            # Temporarily move the reference model to GPU for computation
            device = model_retain_activations.device
            self.ref_model = self.ref_model.to(device)
            
            ref_retain_activations, _ = self.forward_with_cache(
                self.ref_model, retain_inputs, module=self.ref_module, no_grad=True
            )

            # Immediately move the reference model back to CPU
            self.ref_model = self.ref_model.cpu()
            torch.cuda.empty_cache()  # Clear GPU cache

            mask = retain_inputs["labels"] != -100  # Shape: [b, s]
            retain_loss = self.compute_activation_loss(
                model_retain_activations,
                ref_retain_activations.to(model_retain_activations.device),
                mask,
            )
        else:
            retain_loss = super().compute_retain_loss(model, retain_inputs)
        return retain_loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        forget = inputs["forget"]
        forget = forget["forget"]
        forget_inputs = {
            "input_ids": forget["input_ids"],
            "attention_mask": forget["attention_mask"],
            "labels": forget["labels"],
        }

        model_forget_activations, forget_outputs = self.forward_with_cache(
            model, forget_inputs, self.model_module, no_grad=False
        )
        # If multiple datasets or concepts need unlearning, pass the control vector during processing; otherwise, default to a random vector during training.
        control_vec = forget_inputs.get(
            "control_vec", self.get_control_vector(model_forget_activations.shape[-1])
        )
        control_vec = control_vec.to(
            dtype=model_forget_activations.dtype, device=model_forget_activations.device
        )
        control_vec = control_vec.expand_as(model_forget_activations)
        mask = forget_inputs["labels"] != -100  # Shape: [b, s]
        forget_loss = self.compute_activation_loss(
            model_forget_activations, control_vec, mask
        )

        retain = inputs["retain"]
        retain = retain["retain"]
        retain_inputs = {
            "input_ids": retain["input_ids"],
            "attention_mask": retain["attention_mask"],
            "labels": retain["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss

        return (loss, forget_outputs) if return_outputs else loss
