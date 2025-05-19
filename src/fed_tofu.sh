export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

models=(
    "Llama-3.2-1B-Instruct"
    "Llama-3.2-3B-Instruct"
    "Llama-3.1-8B-Instruct"
)
splits=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)
aggregation_strategys=(
    "FedAvg"
    "FedAdam"
    "FedYogi"
    "FedAdagrad"
    "FedAvgM"
)
unlearn_trainer_cls_es=(
    "GradAscent"
    "GradDiff"
    "NPO"
    "SimNPO"
    "RMU"
)

per_device_train_batch_size=4 # on two gpus would make effective batch size 32
gradient_accumulation_steps=4

########################################################################################################################
########################################### Federated Unlearn TOFU Models ##############################################
########################################################################################################################

for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)

    for model in "${models[@]}"; do
        for aggregation_strategy in "${aggregation_strategys[@]}"; do
            for unlearn_trainer_cls in "${unlearn_trainer_cls_es[@]}"; do
                task_name=tofu_${model}_${forget_split}_${unlearn_trainer_cls}_${aggregation_strategy}
                model_path=open-unlearning/tofu_${model}_full
                echo ${task_name}: Federated Unlearning ${model_path} using ${unlearn_trainer_cls} with ${aggregation_strategy}

                # Federated Unlearn
                python src/fed_train.py --config-name=unlearn.yaml \
                experiment=unlearn/tofu/default.yaml \
                trainer=FederatedUnlearningTrainer \
                task_name=${task_name} \
                model=${model} \
                forget_split=${forget_split} \
                retain_split=${retain_split} \
                model.method_args.aggregation_strategy=${aggregation_strategy} \
                model.method_args.unlearn_trainer_cls=${unlearn_trainer_cls} \
                model.model_args.pretrained_model_name_or_path=${model_path} \
                retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
                trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
                trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
                trainer.args.ddp_find_unused_parameters=true \
                trainer.args.gradient_checkpointing=true

                # Eval
                CUDA_VISIBLE_DEVICES=0 python src/eval.py \
                experiment=eval/tofu/default.yaml \
                forget_split=${forget_split} \
                holdout_split=${holdout_split} \
                model=${model} \
                task_name=${task_name} \
                model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
                paths.output_dir=saves/unlearn/${task_name}/evals \
                retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json
            done
        done
    done
done