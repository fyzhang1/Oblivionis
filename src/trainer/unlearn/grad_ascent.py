from trainer.unlearn.base import UnlearnTrainer


class GradAscent(UnlearnTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        outputs = model(**forget_inputs)
        loss = -outputs.loss
        return (loss, outputs) if return_outputs else loss

# class GradAscent(UnlearnTrainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         """
#         Compute loss for gradient ascent (unlearning).
#         Expects inputs to be a dict with 'input_ids', 'labels', 'attention_mask'.
#         """
#         # 直接使用 inputs，移除 'forget' 键的提取
#         forget_inputs = {
#             "input_ids": inputs["input_ids"],
#             "attention_mask": inputs["attention_mask"],
#             "labels": inputs.get("labels", None),  # labels 可能为 None
#         }
        
#         # 调用模型计算输出
#         outputs = model(**forget_inputs)
#         loss = outputs.loss
        
#         # 梯度上升：将损失取负
#         loss = -loss
        
#         return (loss, outputs) if return_outputs else loss