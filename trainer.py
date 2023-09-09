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
        scheduler,
        device
    ):
        self.args = args
        self.train_ds = train_ds
        self.test_ds = test_ds
        
        self.device = device
        self.model = model
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.steps = 0
        self.train_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'train'))
        self.valid_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'valid'))
        self.best_acc = 0
    
    
    def update_tensorboard(self, loss, acc, mode="train"):
        if mode == "train":
            self.train_writer.add_scalar("Loss/train", loss, self.steps)
            self.train_writer.add_scalar("Accuracy/train", acc, self.steps)
        elif mode == "valid":
            self.valid_writer.add_scalar("Loss/valid", loss, self.steps)
            self.valid_writer.add_scalar("Accuracy/valid", acc, self.steps)
    
    
    def valid(self):
        test_corr_cnt = 0
        test_loader = DataLoader(self.test_ds, self.args.test_batch_size, shuffle=False)
        
        self.model.eval() # nullify dropout
        with torch.no_grad():
            for step, batch in enumerate(test_loader, 1):
                batch = [b.to(self.device) for b in batch]
                img, label = batch
                y_pred = self.model(img)
                _, predicted = torch.max(y_pred, dim=1)
                test_corr_cnt += (predicted == label).sum().detach().cpu()
                
        loss = self.loss_func(y_pred, label)
        accuracy = 100 * (test_corr_cnt / (step * self.args.test_batch_size))
        
        self.update_tensorboard(
            loss=loss.detach().cpu().item(),
            acc=accuracy.item(),
            mode="valid"
        )
        
        # Save Best Accuracy
        if accuracy > self.best_acc:
            self.best_acc = accuracy.item()
            
            # Save Best Model Checkpoint
            torch.save({
                "steps": self.steps,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "accuracy": accuracy.item(),
                "loss": loss.item()
            }, os.path.join(self.args.ckpt_dir, 'best-model.ckpt'))
    
    
    def train(self):
        self.model.train()
        
        for epoch in tqdm(range(1, self.args.num_epochs+1)):
            train_corr_cnt = 0
            train_loader = DataLoader(self.train_ds, self.args.train_batch_size, shuffle=True)
            
            for step, batch in enumerate(train_loader, 1):
                batch = [b.to(self.device) for b in batch]
                img, label = batch
                y_pred = self.model(img)
                loss = self.loss_func(y_pred, label)
                
                _, prediction = torch.max(y_pred, dim=1)
                train_corr_cnt += (prediction == label).sum().detach().cpu()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.steps += 1
                
                if step % self.args.logging_steps == 0:
                    acc = (train_corr_cnt / (step * self.args.train_batch_size)) * 100
                    print(f"Epoch:{epoch:2d} Batch:{step:2d} Loss:{loss:4.4f} Accuracy:{acc:4.4f}%")
                    
                    self.update_tensorboard(
                        loss=loss.detach().cpu().item(),
                        acc=acc.item(),
                        mode="train"
                    )

                # Save Latest Model Checkpoint
                if step % self.args.save_steps == 0:
                    torch.save({
                        "epochs": epoch,
                        "steps": step,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict()
                    }, os.path.join(self.args.ckpt_dir, "lastest-model.ckpt"))
                
            self.scheduler.step()
            
            # validation
            self.valid()
            
        self.train_writer.flush()
        self.valid_writer.flush()
        self.train_writer.close()
        self.valid_writer.close()