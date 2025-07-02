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
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute loss for gradient ascent (unlearning).
        Expects inputs to be a dict with 'input_ids', 'labels', 'attention_mask'.
        """

        forget = inputs["forget"]
        forget = forget["forget"]
        # Directly use input data since ForgetRetainDataset now returns the required data format
        forget_inputs = {
            "input_ids": forget["input_ids"],
            "attention_mask": forget["attention_mask"],
            "labels": forget.get("labels", None),  # labels 可能为 None
        }
        
        # Call model to compute outputs
        outputs = model(**forget_inputs)
        loss = outputs.loss
        
        # Gradient ascent: negate the loss
        loss = -loss
        
        return (loss, outputs) if return_outputs else loss

    