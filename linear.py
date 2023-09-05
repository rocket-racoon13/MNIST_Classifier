import math
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter


class LinearLayer(nn.Module):
    """
    Role:
    1) Applies linear transformation to the incoming data
    2) stores weight and bias (in a trainable state)
    3) initializes weight and bias
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.bias = None
        self.initialize_parameters()
    
    
    def initialize_parameters(self) -> None:
        self.weight = init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
    
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            result = torch.matmul(input, self.weight) + self.bias
        else:
            result = torch.matmul(input, self.weight)
        return result
    
    
    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        return self.forward(input)