import math
import torch
import torch.optim as optim
from torch.nn import init
from torch.nn.parameter import Parameter

from model import *


class LinearLayer:
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


def get_model(args):
    model_type_name = args.model_type.lower()
    if model_type_name == "ann":
        model = MNISTClassifierANN(args)
    elif model_type_name == "cnn":
        model = MNISTClassifierCNN(args)
    return model
    

def get_optimizer(args, model):
    optim_name = args.optimizer.lower()
    
    if optim_name == "adagrad":
        optimizer = optim.Adagrad(
            params=model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            eps=args.eps
        )
    elif optim_name == "adam":
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            eps=args.eps
        )
    elif optim_name == "sgd":
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    else:
        raise NotImplementedError
    
    return optimizer

        
def get_scheduler(args, optimizer):
    scheduler_name = args.scheduler.lower()
    
    if scheduler_name == "lambdalr":
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda epoch: 0.95**epoch
        )
    elif scheduler_name == "steplr":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=10,
            gamma=0.5
        )
    else:
        raise NotImplementedError
    
    return scheduler


if __name__ == "__main__":
    m = LinearLayer(20, 30)
    input = torch.randn(128, 20)
    output = m(input)
    print(output.size())