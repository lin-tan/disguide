# Helper class for having ensemble models.

import torch
import torch.nn as nn
import torch.nn.functional as F
import network


class Ensemble(nn.Module):
    """Wrapper modeule for an ensemble of clone models"""

    def __init__(self, ensemble_size: int = 16, num_classes: int = 10, student_model: str = "ensemble_resnet18_8x"):
        super(Ensemble, self).__init__()
        if student_model == 'ensemble_resnet18_8x':
            self.subnets = nn.ModuleList([network.resnet_8x.ResNet18_8x(num_classes) for i in range(ensemble_size)])
        elif student_model == 'ensemble_lenet5_half':
            self.subnets = nn.ModuleList([network.lenet.LeNet5Half() for i in range(ensemble_size)])
        else:
            raise NotImplementedError("Only supporting lenet5Half and Resnet18")
    
    def forward(self, x, idx: int = -1):
        if idx >= 0:
            return self.subnets[idx].forward(x)
        results = []
        for i in range(len(self.subnets)):
            results.append(self.subnets[i].forward(x))
        return torch.stack(results, dim=1)
    
    def variance(self, x):
        results = []
        with torch.no_grad():
            for i in range(len(self.subnets)):
                results.append(self.subnets[i].forward(x))
            return torch.var(F.softmax(torch.stack(results, dim=1), dim=-1), dim=1)
    
    def size(self):
        return len(self.subnets)
    
    def get_model_by_idx(self, idx):
        return self.subnets[idx]
        
