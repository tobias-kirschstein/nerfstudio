import torch


class GenericScheduler(torch.nn.Module):
    """A generic scheduler"""

    def __init__(self, final_value, max_step) -> None:
        super().__init__()
        self.value = final_value
        self.final_value = final_value
        self.max_step = max_step

    def update(self, step):
        if step > self.max_step:
            self.value = self.final_value
        else:
            self.value = min(max(step / self.max_step, 0), 1) * self.final_value

    def get_value(self):
        if self.training:  # inherit from torch.nn.Module
            return self.value
        else:
            return self.final_value
