from trainer.unlearn.base import UnlearnTrainer
from data.unlearn import ForgetRetainDataset


# class GradAscent(UnlearnTrainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         forget_inputs = inputs["forgets"]
#         forget_inputs = {
#             "input_ids": forget_inputs["input_ids"],
#             "attention_mask": forget_inputs["attention_mask"],
#             "labels": forget_inputs["labels"],
#         }
#         outputs = model(**forget_inputs)
#         loss = -outputs.loss
#         return (loss, outputs) if return_outputs else loss

class GradAscent(UnlearnTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss for gradient ascent (unlearning).
        Expects inputs to be a dict with 'input_ids', 'labels', 'attention_mask'.
        """

        
        # 非常抱歉，这里我写的非常屎,inputs数据被套了两个forget循环，在这里给他解开即可正常运行，我太懒了
        inputs = inputs["forget"]
        inputs = inputs["forget"]
        # print(inputs)
        forget_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": inputs.get("labels", None),  # labels 可能为 None
        }
        
        # 调用模型计算输出
        outputs = model(**forget_inputs)
        loss = outputs.loss
        
        # 梯度上升：将损失取负
        loss = -loss
        
        return (loss, outputs) if return_outputs else loss

    