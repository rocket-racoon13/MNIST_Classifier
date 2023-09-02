import torch
import torch.optim as optim
from torch.nn.parameter import Parameter


def get_optimizer(args, model):
    optim_name = args.optimizer.lower()
    
    if optim_name == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), args.learning_rate)
    elif optim_name == "adam":
        optimizer = optim.Adam(model.parameters(), args.learning_rate)
    elif optim_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), args.learning_rate)
    elif optim_name == "sgd":
        optimizer = optim.SGD(model.parameters(), args.learning_rate)
        
        
def get_scheduler(args, optimizer):
    scheduler_name = args.scheduler.lower()
    
    if scheduler_name == "steplr":
        pass
    elif scheduler_name == "cosineannealinglr":
        pass