import argparse

import torch
import torch.nn as nn

from dataset import MNIST
from model import MNISTClassifierCNN
from train import *
from test import *
from utils import *

random_seed = 101
train_batch_size = 100
valid_batch_size = 500
test_batch_size = 10000
num_epochs = 10
in_channels = 1
in_features = 784
out_features = 10
lr = 1e-3


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=77)
    parser.add_argument('--data_dir', type=str, default='data/mnist')
    parser.add_arumgnet('--model_dir', type=str)
    
    parser.add_argument('--img_width', type=int, default=28)
    parser.add_argument('--img_height', type=int, default=28)
    parser.add_argument('--img_channel', type=int, default=1)
    parser.add_argument('--conv_channels', type=list, default=[32, 64])
    parser.add_argument('--fc_dims', type=list, default=[128])
    
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=10)
    parser.add_argument('--test_batch_size', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--optimizer')
    parser.add_argument('--scheduler')
    
    parser.add_argument('--logging_steps', type=int, default=200)
    parser.add_argument('--model_save_steps', type=int, default=200)
    
    parser.add_argument('--train', action="store")
    parser.add_argument('--test', action="store")
    
    args = parser.parse_args()
    
    # set random seed
    torch.manual_seed(random_seed)
    
    # load and normalize dataset
    def image_transform(image_np):
        output = to_tensor(image_np, normalize=True)
        # output = output.reshape(output.size(0), -1) # ANN용 reshaper
        output = normalize(output, 0.5, 0.5) # normalize to [-1.0, 1.0] range
        return output
    
    def label_transform(label_np):
        output = to_tensor(label_np, normalize=False, dtype=torch.int64)
        return output
        
    mnist_train = MNIST("dataset/mnist", train=True, transform=image_transform, target_transform=label_transform)
    mnist_test = MNIST("dataset/mnist", train=False, transform=image_transform, target_transform=label_transform)
    
    # create CNN model, loss_func, optimizer
    model = MNISTClassifierCNN(in_channels, out_features)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    
    print(model.summary())
    
    # train
    trainer = Trainer(
        train_ds=mnist_train,
        test_ds=mnist_test,
        train_batch_size=train_batch_size,
        valid_batch_size=valid_batch_size,
        num_epochs=num_epochs,
        model=model,
        loss_func=loss_func,
        optimizer=optimizer)
    trainer.train()
    
    # test
    tester = Tester(
        test_ds=mnist_test,
        test_batch_size=test_batch_size,
        model=trainer.model,
        loss_func=trainer.loss_func,
        optimizer=trainer.optimizer)
    tester.test()
    
    