import torch.nn.functional as F

from trainer.utils import compute_batch_nll
from trainer.unlearn.grad_diff import GradDiff


class SimNPO(GradDiff):
    def __init__(self, delta=0.0, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = delta
        self.beta = beta

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        forget = inputs["forget"]
        forget = forget["forget"]

        forget_labels = forget["labels"]
        loss_mask = forget_labels != -100
        forget_loss, forget_outputs = compute_batch_nll(model, forget)
        forget_loss = forget_loss / loss_mask.sum(-1) - self.delta
        forget_loss = -F.logsigmoid(self.beta * forget_loss).mean() * 2 / self.beta

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
