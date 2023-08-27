from tqdm import tqdm
import torch
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        train_ds,
        test_ds,
        train_batch_size,
        valid_batch_size,
        num_epochs,
        model,
        loss_func,
        optimizer
    ):
        
        self.train_ds = train_ds
        self.test_ds = test_ds
        
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer

        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.num_epochs = num_epochs
        
        self.records = {
            "Loss": {
                "train": [],
                "test": []
            },
            "Accuracy": {
                "train": [],
                "test": []
            }
        }
    
    def valid(self):
        test_corr_cnt = 0
        test_loader = DataLoader(self.test_ds, self.valid_batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch_idx, (img, label) in enumerate(test_loader, 1):
                y_pred = self.model(img)
                _, predicted = torch.max(y_pred, dim=1)
                test_corr_cnt += (predicted == label).sum()
                
        loss = self.loss_func(y_pred, label)
        self.records["Loss"]["test"].append(loss)
        accuracy = 100 * (test_corr_cnt / (batch_idx * self.valid_batch_size))
        self.records["Accuracy"]["test"].append(accuracy.item())
    
    def train(self):
        for epoch in tqdm(range(self.num_epochs)):
            train_corr_cnt = 0
            train_loader = DataLoader(self.train_ds, self.train_batch_size, shuffle=True)
            
            for batch_idx, (img, label) in enumerate(train_loader, 1):
                y_pred = self.model(img)
                loss = self.loss_func(y_pred, label)
                
                _, prediction = torch.max(y_pred, dim=1)
                train_corr_cnt += (prediction == label).sum()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if batch_idx % 200 == 0:
                    acc = (train_corr_cnt / (batch_idx * self.train_batch_size)) * 100
                    print(f"Epoch:{epoch:2d} Batch:{batch_idx:2d} Loss:{loss:4.4f} Accuracy:{acc:4.4f}%")
        
            self.records["Loss"]["train"].append(loss.item())
            accuracy = 100 * (train_corr_cnt / (batch_idx * self.train_batch_size))
            self.records["Accuracy"]["train"].append(accuracy.item())
            
            # validation
            self.valid()