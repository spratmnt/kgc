from typing import Tuple
import torch
from torch import nn
from torch.autograd import Function as F
#from geoopt.manifolds.symmetric_positive_definite import SymmetricPositiveDefinite
from geoopt.linalg import batch_linalg as lalg
from gyromatman.manifolds.metrics import Metric, MetricType
from gyromatman.utils import get_logging, gr_identity
from gyromatman.trivializations import expm
from gyromatman.config import DEVICE

def cayley_map(X):
    n = X.size(-1)
    Id = torch.eye(n, dtype=X.dtype, device=X.device)
    return (Id - X) @ torch.inverse(Id + X)

class GRManifold(nn.Module):
    def __init__(self, dims=2, pdim=2, metric=MetricType.RIEMANNIAN):
        super().__init__()
        self.dims = dims
        self.pdim = pdim
        self.metric = Metric.get(metric.value, self.dims)

        base_point = gr_identity(dims, pdim) 
        self.register_buffer('base_point', base_point)

    def dist(self, a: torch.Tensor, b: torch.Tensor, keepdim=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute distance between 2 points on the Grassmann manifold        

        :param a, b: b x n x n: elements in Grassmann
        :return: distance: b: torch.Tensor with distances between a and b in Grassmann
        :return: vvd: b x n: torch.Tensor with vector-valued distance between a and b
        """                

        outputs = torch.norm(a - b, dim=(-1,-2)).unsqueeze(-1)

        return outputs, 0 

    def dist_eval(self, a: torch.Tensor, b: torch.Tensor, keepdim=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute distance between 2 points on the Grassmann manifold        

        :param a, b: b x n x n: elements in Grassmann
        :return: distance: b: torch.Tensor with distances between a and b in Grassmann
        :return: vvd: b x n: torch.Tensor with vector-valued distance between a and b
        """                

        a = a.unsqueeze(1)
        return self.dist(a,b)        

    def expmap_id(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs an exponential map using the Identity as basepoint :math:`\operatorname{Exp}_{Id}(x)`.
        :param: x: b x n x n torch.Tensor skew-symmetric matrices
        """        
        
        outputs = torch.matmul( cayley_map(x), torch.matmul(self.base_point, cayley_map(-x)) )
        return outputs    

    def addition_id_from_skew(self, skew_a: torch.Tensor, skew_b: torch.Tensor):
        """
        Performs addition using the Identity as basepoint.        

        The addition on Grassmann using the identity as basepoint is :math:`A \oplus_{Id} B = expm(A) @ expm(B) @ Inp @ expm(-B) @ expm(-A)`.
        
        """        
        skew_ab_positive = torch.matmul(cayley_map(skew_a),  cayley_map(skew_b))
        skew_ba_negative = torch.matmul(cayley_map(-skew_b), cayley_map(-skew_a))
        outputs = torch.matmul( skew_ab_positive, torch.matmul(self.base_point, skew_ba_negative) )
        return outputs    
