from typing import Tuple
import torch
from geoopt.manifolds.symmetric_positive_definite import SymmetricPositiveDefinite
from geoopt.linalg import batch_linalg as lalg
from gyromatman.manifolds.metrics import Metric, MetricType
from gyromatman.utils import get_logging


class SPDLEManifold(SymmetricPositiveDefinite):
    def __init__(self, dims=2, ndim=2, metric=MetricType.RIEMANNIAN):
        super().__init__()
        self.dims = dims
        self.ndim = ndim
        self.metric = Metric.get(metric.value, self.dims)

    def dist(self, a: torch.Tensor, b: torch.Tensor, keepdim=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute distance between 2 points on the SPD manifold        

        :param a, b: b x n x n: elements in SPD
        :return: distance: b: torch.Tensor with distances between a and b in SPD
        :return: vvd: b x n: torch.Tensor with vector-valued distance between a and b
        """        
        d_log = a - b
        result = torch.norm(d_log, dim=(-1,-2)).unsqueeze(-1)

        return result, 0

    def dist_eval(self, a: torch.Tensor, b: torch.Tensor, keepdim=True) -> Tuple[torch.Tensor, torch.Tensor]:
        a = a.unsqueeze(1)
        return self.dist(a,b)       

    @staticmethod
    def expmap_id(x: torch.Tensor) -> torch.Tensor:
        """
        Performs an exponential map using the Identity as basepoint :math:`\operatorname{Exp}_{Id}(u)`.
        :param: x: b x n x n torch.Tensor point on the SPD manifold
        """
        return lalg.sym_funcm(x, torch.exp)

    @staticmethod
    def logmap_id(y: torch.Tensor) -> torch.Tensor:
        """
        Perform an logarithmic map using the Identity as basepoint :math:`\operatorname{Log}_{Id}(y)`.
        :param: y: b x n x n torch.Tensor point on the tangent space of the SPD manifold
        """
        return lalg.sym_funcm(y, torch.log)    

    @staticmethod
    def addition_id_from_log(log_a: torch.Tensor, log_b: torch.Tensor):
        """
        Performs addition using the Identity as basepoint.
        Assumes that sqrt_a = sqrt(A) so it does not apply the sqrt again

        The addition on SPD using the identity as basepoint is :math:`A \oplus_{Id} B = sqrt(A) B sqrt(A)`.

        :param sqrt_a: b x n x n torch.Tensor points in the SPD manifold
        :param b: b x n x n torch.Tensor points in the SPD manifold.
        :return: b x n x n torch.Tensor points in the SPD manifold
        """

        return log_a + log_b    
