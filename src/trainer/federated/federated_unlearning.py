import torch
import logging
from copy import deepcopy
from typing import Dict, List, Any, Optional
import os
from trainer.base import FinetuneTrainer
from trainer.unlearn.base import UnlearnTrainer
from data.unlearn import ForgetRetainDataset
from torch.utils.data import Dataset
from trainer.federated.federated_utils import *
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict


logger = logging.getLogger(__name__)

class FederatedUnlearningTrainer(FinetuneTrainer):

    
    def __init__(
        self,
        model,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        data_collator=None,
        args=None,
        evaluator=None,
        template_args=None,
        num_clients=3,
        target_client_idx=0,
        unlearn_trainer_cls=None,
        aggregation_strategy="average",
        global_rounds=1,
        unlearn_epochs=None,
        **kwargs
    ):

        self.federated_dataset = train_dataset if isinstance(train_dataset, dict) else None
        dummy_train_dataset = None
        if self.federated_dataset and 0 in self.federated_dataset:
            dummy_train_dataset = self.federated_dataset[0].retain
            # dummy_train_dataset = self.federated_dataset[0].get("retain")
        
        super().__init__(
            model=model,
            train_dataset=dummy_train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            args=args,
            evaluator=evaluator,
            template_args=template_args,
        )
        
        # logger.info(f"Training args: {self.args}")
        self.num_clients = num_clients
        self.target_client_idx = target_client_idx
        self.unlearn_trainer_cls = unlearn_trainer_cls
        self.aggregation_strategy = aggregation_strategy
        self.global_rounds = global_rounds
        self.unlearn_epochs = unlearn_epochs  
        self.client_models = []
        self.client_data_sizes = [] 
        self.kwargs = kwargs
        
  
        self.is_peft = isinstance(self.model, PeftModel)

        self.server_momentum = None  
        self.server_velocity = None  
        
        
        self.fedadam_momentum = None  
        self.fedadam_velocity = None   

        self.fedyogi_momentum = None  
        self.fedyogi_velocity = None   
        

        self.fedavgm_momentum = None   
        
 
        self.fedadagrad_velocity = None  
        

        self.federated_args = {
            'server_lr': kwargs.get('server_lr', 1.0),
            'beta1': kwargs.get('beta1', 0.9),
            'beta2': kwargs.get('beta2', 0.99),
            'epsilon': kwargs.get('epsilon', 1e-3),
            'tau': kwargs.get('tau', 1e-3),
            'mu': kwargs.get('mu', 0.01),
            'momentum_factor': kwargs.get('momentum_factor', 0.9)
        }
        
        if not self.federated_dataset:
            logger.warning("No federated dataset provided!")
        else:
            for client_idx in range(num_clients):
                if client_idx not in self.federated_dataset:
                    logger.warning(f"Client {client_idx} has no data!")
        
        logger.info(f"Initialized with {num_clients} clients, target: {target_client_idx}, global rounds: {global_rounds}")
        logger.info(f"Model type - PEFT: {self.is_peft}")


    def train(self, resume_from_checkpoint=None, trial=None, **kwargs):
        if not self.federated_dataset:
            raise ValueError("Federated dataset is not properly set up")
        
        logger.info(f"Starting federated unlearning training with {self.global_rounds} global rounds")
        
     
        self.model = self.model.cpu()
        self.model = deepcopy(self.model)
        

       
        logger.info(f"Step 1: Unlearning on target client {self.target_client_idx}")
        if self.target_client_idx in self.federated_dataset:
            target_dataset = self.federated_dataset[self.target_client_idx]
         
            self.unlearn_client_model(self.model, target_dataset)            
            
           
         
            logger.info("Updated global model with unlearned model")
        else:
            logger.warning(f"Target client {self.target_client_idx} has no data, skipping unlearning")
    
        for round_idx in range(self.global_rounds):
            logger.info(f"Starting global round {round_idx + 1}/{self.global_rounds}")
            
    
            logger.info("Step 2: All clients training on retain data with updated global model")
            
         
            self.client_models = [deepcopy(self.model) for _ in range(self.num_clients)]
            self.client_data_sizes = [] 

           
            logger.info(f"Aggregation strategy: {self.aggregation_strategy}")


            for client_idx in range(self.num_clients):
                if client_idx not in self.federated_dataset:
                    logger.warning(f"Skipping client {client_idx} as no data is available")
                    self.client_data_sizes.append(0)
                    continue
                    
                client_model = self.client_models[client_idx]
                client_dataset = self.federated_dataset[client_idx]
                
                retain_data = client_dataset.retain.retain
                client_data_size = len(retain_data)
                self.client_data_sizes.append(client_data_size)
                
                logger.info(f"Training client {client_idx} on retain data with {self.args.num_train_epochs} epochs")
                self.train_client_model(client_model, client_dataset)
                
                self.client_models[client_idx] = client_model.cpu()

            logger.info("Step 3: Aggregating all client models")
            self.aggregate_models(round_idx)
            logger.info(f"Completed global round {round_idx + 1}/{self.global_rounds}")
            
            if self.args.save_strategy == "epoch" or (round_idx == self.global_rounds - 1):
                save_path = f"{self.args.output_dir}/round_{round_idx + 1}"
                self.model = self.model.cpu()
                self.save_model(save_path)
                logger.info(f"Model saved at {save_path}")
        
        self.save_state()
        return None

    def unlearn_client_model(self, client_model, client_dataset):
        from trainer import TRAINER_REGISTRY
        trainer_cls = TRAINER_REGISTRY.get(self.unlearn_trainer_cls, None)


        forget_dataset = client_dataset.forget
        retain_dataset = client_dataset.retain


        logger.info(f"forget_data length: {len(forget_dataset)}")
        logger.info(f"retain_data length: {len(retain_dataset.retain)}")
        logger.info(f"Unlearning_cls_name: {self.unlearn_trainer_cls}")
        
        class CombinedDataset(Dataset):
            def __init__(self, forget_data, retain_data):
                self.forget_data = forget_data
                self.retain_data = retain_data
                self.retain_length = len(retain_data)

            def __len__(self):
                return len(self.forget_data)

            def __getitem__(self, idx):
                retain_idx = idx % self.retain_length
                return {
                    "forget": self.forget_data[idx],
                    "retain": self.retain_data[retain_idx]
                }
        
        combined_dataset = CombinedDataset(forget_dataset, retain_dataset)

        unlearn_args = deepcopy(self.args)
        if self.unlearn_epochs is not None:
            unlearn_args.num_train_epochs = self.unlearn_epochs
            logger.info(f"Using {self.unlearn_epochs} epochs for unlearn stage")
        else:
            logger.info(f"Using default {unlearn_args.num_train_epochs} epochs for unlearn stage")

        if self.aggregation_strategy == "FedProx":
            class FedProxUnlearnTrainer(trainer_cls):
                def __init__(self, *args, global_model=None, mu=0.01, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.global_model = global_model.to(self.args.device) if global_model is not None else None
                    self.mu = mu

                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    return_values = super().compute_loss(model, inputs, return_outputs=return_outputs)
                    
                    if return_outputs:
                        loss, outputs = return_values
                    else:
                        loss = return_values

                    proximal_term = 0.0
                    if self.global_model is not None:
                        global_state = get_peft_model_state_dict(self.global_model)
                        for name, w in model.named_parameters():
                            if not w.requires_grad:
                                continue
                            name = name.replace(".default", "") 
                            if name not in global_state:
                                logger.warning(f"Parameter {name} not found in global_state, skipping")
                                continue
                            proximal_term += torch.norm(w - global_state[name].to(w.device).detach()) ** 2
                        loss += (self.mu / 2) * proximal_term
                    
                    return (loss, outputs) if return_outputs else loss
            
            unlearn_trainer = FedProxUnlearnTrainer(
                model=client_model,
                train_dataset=combined_dataset,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                args=unlearn_args, 
                evaluator=self.evaluator,
                template_args=self.template_args,
                global_model=self.model.cpu(),
                mu=self.federated_args['mu'],
                **self.kwargs
            )
        else:
            unlearn_trainer = trainer_cls(
                model=client_model,
                train_dataset=combined_dataset,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                args=unlearn_args, 
                evaluator=self.evaluator,
                template_args=self.template_args,
                **self.kwargs
            )
        

        unlearn_trainer.train()
        
        return unlearn_trainer
    

    def train_client_model(self, client_model, client_dataset):
        retain_dataset = client_dataset.retain
        retain_data = retain_dataset.retain
        
        if self.aggregation_strategy == "FedProx":
            class FedProxTrainer(FinetuneTrainer):
                def __init__(self, *args, global_model=None, mu=0.01, **kwargs):
                    super(FedProxTrainer, self).__init__(*args, **kwargs)
                    self.global_model = global_model
                    self.mu = mu
                    
                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    return_values = super().compute_loss(model, inputs, return_outputs=return_outputs)
                    
                    if return_outputs:
                        loss, outputs = return_values
                    else:
                        loss = return_values
                    proximal_term = 0.0
                    if self.global_model is not None:
                        global_state = get_peft_model_state_dict(self.global_model)
                        for name, w in model.named_parameters():
                            if not w.requires_grad:
                                continue
                            name = name.replace(".default", "") 
                            if name not in global_state:
                                logger.warning(f"Parameter {name} not found in global_state, skipping")
                                continue
                            proximal_term += torch.norm(w - global_state[name].to(w.device).detach()) ** 2
                        loss += (self.mu / 2) * proximal_term                    
                    return (loss, outputs) if return_outputs else loss
                
            trainer = FedProxTrainer(
                model=client_model,
                train_dataset=retain_data,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                args=self.args,
                evaluator=self.evaluator,
                template_args=self.template_args,
                global_model=self.model.cpu(),
                mu=self.federated_args['mu']
            )
        else:
            trainer = FinetuneTrainer(
                model=client_model,
                train_dataset=retain_data,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                args=self.args,
                evaluator=self.evaluator,
                template_args=self.template_args,
            )
        
        trainer.train()
        return trainer
    

    def aggregate_models(self, round_idx=0):
        logger.info("Aggregating client models")
        

        total_samples = sum(self.client_data_sizes)
        if total_samples == 0:
            logger.warning("No samples available for aggregation")
            return
            
        client_weights = [size / total_samples for size in self.client_data_sizes]
        logger.info(f"Client sample sizes: {self.client_data_sizes}")
        logger.info(f"Client weights: {[f'{w:.4f}' for w in client_weights]}")

        if self.is_peft:
            client_state_dicts = []
            for model in self.client_models:

                # model = model.cpu()
                # peft_state_dict = model.get_peft_model_state_dict()
                peft_state_dict = get_peft_model_state_dict(model.cpu())
                client_state_dicts.append(peft_state_dict)
                logger.debug(f"Client {model} has PEFT state dict")
        else:

            client_state_dicts = [model.cpu().state_dict() for model in self.client_models]

        logger.info(f"Aggregation_strategy:{self.aggregation_strategy}")
        
        if self.is_peft:
            # global_peft_state_dict = self.model.cpu().get_peft_model_state_dict()
            global_peft_state_dict = get_peft_model_state_dict(self.model.cpu())
        else:
            global_peft_state_dict = self.model.cpu().state_dict()

        if self.aggregation_strategy == "FedAvg":
            global_state_dict = FedAvg(
                client_state_dicts, 
                global_model_state_dict=global_peft_state_dict,
                client_weights=client_weights
            )
        elif self.aggregation_strategy == "FedAvgM":
            global_state_dict, self.fedavgm_momentum = FedAvgM(
                client_state_dicts,
                global_model_state_dict=global_peft_state_dict,
                proxy_dict=self.fedavgm_momentum,
                round_idx=round_idx,
                momentum_factor=self.federated_args['momentum_factor'],
                tau=self.federated_args['tau'],
                client_weights=client_weights
            )
        elif self.aggregation_strategy == "FedAdagrad":
            global_state_dict, self.fedadagrad_velocity = FedAdagrad(
                client_state_dicts,
                global_model_state_dict=global_peft_state_dict,
                server_velocity=self.fedadagrad_velocity,
                epsilon=self.federated_args['epsilon'],
                tau=self.federated_args['tau'],
                client_weights=client_weights
            )
        elif self.aggregation_strategy == "FedYogi":
            global_state_dict, self.fedyogi_momentum, self.fedyogi_velocity = FedYogi(
                client_state_dicts,
                global_model_state_dict=global_peft_state_dict,
                proxy_dict=self.fedyogi_momentum,
                opt_proxy_dict=self.fedyogi_velocity,
                round_idx=round_idx,
                beta1=self.federated_args['beta1'],
                beta2=self.federated_args['beta2'],
                epsilon=self.federated_args['epsilon'],
                tau=self.federated_args['tau'],
                client_weights=client_weights
            )
        elif self.aggregation_strategy == "FedAdam":
            global_state_dict, self.fedadam_momentum, self.fedadam_velocity = FedAdam(
                client_state_dicts,
                global_model_state_dict=global_peft_state_dict,
                proxy_dict=self.fedadam_momentum,
                opt_proxy_dict=self.fedadam_velocity,
                round_idx=round_idx,
                beta1=self.federated_args['beta1'],
                beta2=self.federated_args['beta2'],
                epsilon=self.federated_args['epsilon'],
                tau=self.federated_args['tau'],
                client_weights=client_weights
            )

        elif self.aggregation_strategy == "FedProx":
            global_state_dict = FedAvg(
                client_state_dicts,
                global_model_state_dict=global_peft_state_dict,
                client_weights=client_weights
            )
            
        if self.is_peft:
            set_peft_model_state_dict(self.model, global_state_dict)
        else:
            self.model.load_state_dict(global_state_dict)

        logger.info("Global model updated")