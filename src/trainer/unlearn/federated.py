import os
import copy
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any

from src.trainer.unlearn.base import UnlearnTrainer

class FederatedUnlearnTrainer(UnlearnTrainer):
    """
    Federated Unlearning Trainer for coordinating the unlearning process across multiple clients.
    """
    
    def __init__(
        self,
        num_clients: int = 5,
        target_client_idx: int = 0,
        client_epochs: int = 1,
        aggregation_strategy: str = "fedavg",
        **kwargs
    ):
        """
        Initialize the federated unlearning trainer.
        
        Args:
            num_clients: Number of clients in federated learning setup
            target_client_idx: Index of the client that will unlearn (forget) data
            client_epochs: Number of epochs to train each client
            aggregation_strategy: Strategy for aggregating client models ('fedavg', 'weighted')
            **kwargs: Additional arguments to pass to the parent UnlearnTrainer
        """
        super().__init__(**kwargs)
        
        self.num_clients = num_clients
        self.target_client_idx = target_client_idx
        self.client_epochs = client_epochs
        self.aggregation_strategy = aggregation_strategy
        self.client_models = {}
        
        # Validate inputs
        if self.target_client_idx >= self.num_clients:
            raise ValueError(f"Target client index {self.target_client_idx} must be less than number of clients {self.num_clients}")
        
        if self.aggregation_strategy not in ["fedavg", "weighted"]:
            raise ValueError(f"Aggregation strategy '{self.aggregation_strategy}' not supported. Choose from: 'fedavg', 'weighted'")
        
        self.logger.info(f"Initialized FederatedUnlearnTrainer with {self.num_clients} clients")
        self.logger.info(f"Target client for unlearning: {self.target_client_idx}")
        self.logger.info(f"Aggregation strategy: {self.aggregation_strategy}")

    def initialize_client_models(self):
        """Initialize models for all clients with the same weights as the global model."""
        global_state_dict = self.model.state_dict()
        
        for client_idx in range(self.num_clients):
            # Create a deep copy of the model for each client
            client_model = copy.deepcopy(self.model)
            client_model.load_state_dict(copy.deepcopy(global_state_dict))
            self.client_models[client_idx] = client_model
            
            # Move client model to the same device as the global model
            self.client_models[client_idx].to(self.args.device)
            
        self.logger.info(f"Initialized {self.num_clients} client models")

    def train_client(self, client_idx, client_dataset, is_unlearning=False):
        """
        Train a client model on its dataset.
        
        Args:
            client_idx: Index of the client to train
            client_dataset: Dataset for this client (dict with 'forget' and 'retain' keys)
            is_unlearning: Whether this client should perform unlearning
            
        Returns:
            Trained client model
        """
        self.logger.info(f"Training client {client_idx} (unlearning={is_unlearning})")
        
        # Get the client model
        client_model = self.client_models[client_idx]
        client_model.train()
        
        # Prepare optimizer for this client
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in client_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in client_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        client_optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
        )
        
        # Training loop
        for epoch in range(self.client_epochs):
            if is_unlearning and client_dataset["forget"] is not None:
                # Apply unlearning on the forget dataset
                self.unlearn_step(client_model, client_dataset["forget"], client_optimizer)
            
            if client_dataset["retain"] is not None:
                # Standard training on the retain dataset
                train_dataloader = self.get_train_dataloader(client_dataset["retain"])
                for batch in train_dataloader:
                    # Move batch to device
                    batch = {k: v.to(self.args.device) if isinstance(v, torch.Tensor) else v 
                             for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = client_model(**batch)
                    loss = outputs.loss
                    
                    # Backward pass
                    loss.backward()
                    client_optimizer.step()
                    client_optimizer.zero_grad()
        
        return client_model

    def unlearn_step(self, model, forget_dataset, optimizer):
        """
        Perform unlearning steps on the target client model.
        This can be overridden in subclasses to implement specific unlearning algorithms.
        
        Args:
            model: The client model to unlearn from
            forget_dataset: Dataset containing samples to forget
            optimizer: Optimizer for the client model
        """
        # Create dataloader for forget dataset
        forget_dataloader = self.get_train_dataloader(forget_dataset)
        
        for batch in forget_dataloader:
            # Move batch to device
            batch = {k: v.to(self.args.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # For unlearning, we maximize the loss (gradient ascent)
            # by negating the gradients
            loss = -loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def aggregate_models(self, client_models, client_data_sizes=None):
        """
        Aggregate client models into a global model.
        
        Args:
            client_models: Dictionary of client models
            client_data_sizes: Dictionary of client data sizes (for weighted averaging)
            
        Returns:
            Aggregated global model
        """
        self.logger.info("Aggregating client models into global model")
        
        # Get the global model's state dict
        global_state_dict = self.model.state_dict()
        
        # Initialize weights for aggregation
        if self.aggregation_strategy == "fedavg":
            # Simple averaging (equal weights)
            weights = {client_idx: 1.0 / len(client_models) for client_idx in client_models}
        elif self.aggregation_strategy == "weighted" and client_data_sizes:
            # Weighted averaging based on dataset sizes
            total_size = sum(client_data_sizes.values())
            weights = {client_idx: size / total_size for client_idx, size in client_data_sizes.items()}
        else:
            # Default to equal weights
            weights = {client_idx: 1.0 / len(client_models) for client_idx in client_models}
        
        # Perform weighted averaging of model parameters
        for key in global_state_dict.keys():
            # Skip aggregation for batch normalization stats for better stability
            if "running_mean" in key or "running_var" in key:
                continue
                
            # Initialize parameter with zeros
            global_state_dict[key] = torch.zeros_like(global_state_dict[key])
            
            # Weighted sum of parameters from all clients
            for client_idx, client_model in client_models.items():
                global_state_dict[key] += weights[client_idx] * client_model.state_dict()[key]
        
        # Load aggregated parameters back to global model
        self.model.load_state_dict(global_state_dict)
        
        return self.model

    def federated_train(self, client_datasets):
        """
        Perform federated training with unlearning on the target client.
        
        Args:
            client_datasets: Dictionary mapping client indices to their datasets
            
        Returns:
            Trained global model with unlearning applied
        """
        self.logger.info("Starting federated training with unlearning")
        
        # Initialize client models with global model weights
        self.initialize_client_models()
        
        # Calculate dataset sizes for weighted averaging (if needed)
        client_data_sizes = {}
        for client_idx, dataset in client_datasets.items():
            # Count samples in retain dataset
            retain_size = len(dataset["retain"]) if dataset["retain"] is not None else 0
            client_data_sizes[client_idx] = retain_size
        
        # Train each client on its local dataset
        for client_idx, client_dataset in client_datasets.items():
            # Check if this is the target client for unlearning
            is_unlearning = (client_idx == self.target_client_idx)
            
            # Train the client
            self.client_models[client_idx] = self.train_client(
                client_idx=client_idx,
                client_dataset=client_dataset,
                is_unlearning=is_unlearning
            )
        
        # Aggregate client models into the global model
        self.aggregate_models(self.client_models, client_data_sizes)
        
        self.logger.info("Completed federated training with unlearning")
        return self.model

    def train(self, train_dataset, eval_dataset=None, **kwargs):
        """
        Override the train method to use federated training with unlearning.
        
        Args:
            train_dataset: Dataset for training (expected to be a federated dataset)
            eval_dataset: Dataset for evaluation
            **kwargs: Additional arguments
            
        Returns:
            Training results
        """
        # Verify train_dataset is in the expected format for federated learning
        if not isinstance(train_dataset, dict):
            raise ValueError("For federated training, train_dataset must be a dictionary mapping client indices to their datasets")
        
        # Perform federated training
        self.federated_train(train_dataset)
        
        # Evaluate the global model if evaluation dataset is provided
        if eval_dataset is not None:
            eval_results = self.evaluate(eval_dataset)
            return {"eval_results": eval_results}
        
        return {} 