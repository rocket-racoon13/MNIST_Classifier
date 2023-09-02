import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(
        self,
        args,
        train_ds,
        test_ds,
        model,
        optimizer,
        scheduler
    ):
        self.args = args
        self.train_ds = train_ds
        self.test_ds = test_ds
        
        self.model = model
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.steps = 0
        self.train_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'train'))
        self.valid_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'valid'))
        self.records = {
            "Loss": {
                "train": [],
                "test": []
            },
            "Accuracy": {
                "train": [],
                "test": [],
                "best": 0
            }
        }
    
    def update_tensorboard(self, loss, acc, epoch=0, mode="train"):
        if mode == "train":
            self.train_writer.add_scalar("Loss/train", loss, self.steps)
            self.train_writer.add_scalar("Accuracy/train", acc, self.steps)
        elif mode == "valid":
            self.valid_writer.add_scalar("Loss/valid", loss, epoch)
            self.valid_writer.add_scalar("Accuracy/valid", acc, epoch)
    
    def valid(self, epoch):
        test_corr_cnt = 0
        test_loader = DataLoader(self.test_ds, self.args.test_batch_size, shuffle=False)
        
        with torch.no_grad():
            for step, (img, label) in enumerate(test_loader, 1):
                y_pred = self.model(img)
                _, predicted = torch.max(y_pred, dim=1)
                test_corr_cnt += (predicted == label).sum()
                
        loss = self.loss_func(y_pred, label)
        self.records["Loss"]["test"].append(loss.item())
        accuracy = 100 * (test_corr_cnt / (step * self.args.test_batch_size))
        self.records["Accuracy"]["test"].append(accuracy.item())
        
        self.update_tensorboard(
            loss=loss.item(),
            acc=accuracy.item(),
            epoch=epoch,
            mode="valid"
        )
        
        # Save Best Accuracy
        if accuracy > self.records["Accuracy"]["best"]:
            self.records["Accuracy"]["best"] = accuracy.item()
            
            # Save Best Model Checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "accuracy": accuracy.item(),
                "loss": loss.item()
            }, os.path.join(self.args.model_dir, 'best-model.ckpt'))
    
    def train(self):
        for epoch in tqdm(range(self.args.num_epochs)):
            train_corr_cnt = 0
            train_loader = DataLoader(self.train_ds, self.args.train_batch_size, shuffle=True)
            
            for step, (img, label) in enumerate(train_loader, 1):
                y_pred = self.model(img)
                loss = self.loss_func(y_pred, label)
                
                _, prediction = torch.max(y_pred, dim=1)
                train_corr_cnt += (prediction == label).sum()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if step % self.args.logging_steps == 0:
                    acc = (train_corr_cnt / (step * self.args.train_batch_size)) * 100
                    print(f"Epoch:{epoch:2d} Batch:{step:2d} Loss:{loss:4.4f} Accuracy:{acc:4.4f}%")
                    
                    self.update_tensorboard(
                        loss=loss.item(),
                        acc=acc.item(),
                        epoch=epoch,
                        mode="train"
                    )

                # Save Latest Model Checkpoint
                if step % self.args.save_steps == 0:
                    torch.save({
                        "epoch": epoch,
                        "steps": step,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict()
                    }, os.path.join(self.args.model_dir, "lastest-model.ckpt"))
                
            self.records["Loss"]["train"].append(loss.item())
            accuracy = 100 * (train_corr_cnt / (step * self.args.train_batch_size))
            self.records["Accuracy"]["train"].append(accuracy.item())
            
            # validation
            self.valid(epoch)