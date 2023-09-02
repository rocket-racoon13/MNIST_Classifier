import argparse

import torch
import torch.nn as nn

from train import *
from test import *
from utils import *
from dataset import MNIST
from model import MNISTClassifierCNN
from model.utils import get_optimizer, get_scheduler


def main(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("gpu") if use_cuda else torch.device("cpu")
    
    set_seed(args)
    
    # load and normalize dataset
    def image_transform(image_np):
        output = to_tensor(image_np, normalize=True)
        # output = output.reshape(output.size(0), -1) # ANN용 reshaper
        output = normalize(output, 0.5, 0.5) # normalize to [-1.0, 1.0] range
        return output
    
    def label_transform(label_np):
        output = to_tensor(label_np, normalize=False, dtype=torch.int64)
        return output
    
    # load dataset
    mnist_train = MNIST("dataset/mnist", train=True,
                        transform=image_transform, target_transform=label_transform)
    mnist_test = MNIST("dataset/mnist", train=False,
                       transform=image_transform, target_transform=label_transform)

    # create model, optimizer, scheduler
    model = MNISTClassifierCNN(args).to(device)
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)
    
    # load model
    if args.model_name is not None:
        ckpt = torch.load(os.path.join(
            args.model_dir, args.model_name
        ))
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        epoch = ckpt["epoch"]
    
    # train
    if args.train:
        trainer = Trainer(
            args,
            train_ds=mnist_train,
            test_ds=mnist_test,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler
        )
        trainer.train()
    
    # test
    if args.test:
        tester = Tester(
            args,
            test_ds=mnist_test,
            model=model,
            optimizer=optimizer
        )
        tester.test()
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=77)
    
    parser.add_argument('--data_dir', type=str, default='dataset/mnist')
    parser.add_arumgnet('--model_dir', type=str, default='model/ckpt')
    parser.add_argument('--log_dir', type=str, default="model/log")
    parser.add_argument('--model_name', type=str, default='best-model.ckpt')
    
    parser.add_argument('--num_labels', type=int, default=10)
    parser.add_argument('--image_width', type=int, default=28)
    parser.add_argument('--image_height', type=int, default=28)
    parser.add_argument('--image_channel', type=int, default=1)
    
    parser.add_argument('--conv_channels', type=list, default=[32, 64])
    parser.add_argument('--fc_dims', type=list, default=[128])
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--padding', type=int, default=1)
    
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--optimizer', type=str, default="AdamW")
    parser.add_argument('--scheduler', type=str, default="StepLR")
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    
    parser.add_argument('--logging_steps', type=int, default=200)
    parser.add_argument('--save_steps', type=int, default=600)
    
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--no_cuda', action="store_true")
    
    args = parser.parse_args()
    
    main(args)