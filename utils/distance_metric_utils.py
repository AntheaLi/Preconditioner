import torch
from torch.nn import Module

class CosineEmbeddingLossModified(Module):
    def __init__(self, reduction='sum'):
        super(CosineEmbeddingLossModified, self).__init__()
        self.loss = torch.nn.CosineEmbeddingLoss(reduction=reduction)
    
    def forward(self, a, b, target=1.0):
        target=torch.tensor([target], dtype=a.dtype).to(a.device)
        result = self.loss(a, b, target=target)
        return result
    

def distance_metric(loss='l2'):
    if loss == 'l2':
        loss_function_u = (torch.nn.MSELoss(reduction='sum'), 1.0)
        loss_function_b = (torch.nn.MSELoss(reduction='sum'), 1.0)
        loss_function_diag = (torch.nn.MSELoss(reduction='sum'), 1.0)
        loss_function_diag2 = (torch.nn.MSELoss(reduction='sum'), 1.0)

    elif loss == 'l1':
        loss_function_u = (torch.nn.L1Loss(reduction='mean'), 1.0)
        loss_function_b = (torch.nn.L1Loss(reduction='mean'), 1.0)
        loss_function_diag = (torch.nn.L1Loss(reduction='mean'), 1.0)
        loss_function_diag2 = (torch.nn.L1Loss(reduction='mean'), 1.0)
    
    elif loss == 'cos':
        loss_function_u = (torch.nn.CosineSimilarity(dim=1), -1.0)
        loss_function_b = (torch.nn.CosineSimilarity(dim=1), -1.0)
        loss_function_diag = (torch.nn.CosineSimilarity(dim=1), -1.0)
        loss_function_diag2 = (torch.nn.CosineSimilarity(dim=1), -1.0)
    

    else:
        raise Exception(f'loss function {loss} is not defined')    

    loss_dict = { 'x': loss_function_u, 
                'b': loss_function_b, 
                'diag': loss_function_diag,
                'diag2': loss_function_diag2}

    return loss_dict