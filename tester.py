from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Tester:
    def __init__(
        self,
        args,
        test_ds,
        model,
        optimizer,
        device
    ):  
        self.args = args
        
        self.test_ds = test_ds
        self.test_batch_size = args.test_batch_size
        
        self.device = device
        self.model = model
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        
        self.records = {
            "Loss": 0,
            "Accuracy": 0
        }
    
    def test(self):
        test_corr_cnt = 0
        test_loader = DataLoader(self.test_ds, self.test_batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch_idx, (img, label) in enumerate(test_loader, 1):
                y_pred = self.model(img)
                _, predicted = torch.max(y_pred, dim=1)
                test_corr_cnt += (predicted == label).sum()
                
        loss = self.loss_func(y_pred, label)
        self.records["Loss"] = loss
        accuracy = 100 * (test_corr_cnt / (batch_idx * self.test_batch_size))
        self.records["Accuracy"] = accuracy
        
        print(f"Test Accuracy: {test_corr_cnt.item()*100/len(self.test_ds):2.4f}%")