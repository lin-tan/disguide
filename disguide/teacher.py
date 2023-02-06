import torch
import torch.nn as nn


class TeacherModel(nn.Module):
    """Wrapper class for teacher model. Handles input transform and teacher network."""
    def __init__(self, model, transform=None):
        super(TeacherModel, self).__init__()
        self.model = model
        self.transform = transform

    def forward(self, x):
        """Handles forward call to teacher network. Guarantees no gradients can be propagated."""
        assert not self.training
        with torch.no_grad():
            if self.transform:
                x = self.transform(x)
            return self.model(x)
