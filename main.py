import argparse
from datetime import datetime

import torch
import torch.nn as nn

from dataset import MNIST
from model_utils import *
from predict import *
from trainer import *
from tester import *
from utils import *


def config():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=77)
    
    parser.add_argument('--data_dir', type=str, default='dataset/mnist')
    parser.add_argument('--ckpt_dir', type=str, default=f"outputs/{datetime.now().strftime('%Y%m%d_%H-%M-%S')}/ckpt")
    parser.add_argument('--log_dir', type=str, default=f"outputs/{datetime.now().strftime('%Y%m%d_%H-%M-%S')}/log")
    parser.add_argument('--ckpt_name', type=str, default="outputs/20230905_22-49-20/ckpt/best-model.ckpt")
    
    parser.add_argument('--num_labels', type=int, default=10)
    parser.add_argument('--image_width', type=int, default=28)
    parser.add_argument('--image_height', type=int, default=28)
    parser.add_argument('--image_channel', type=int, default=1)
    
    parser.add_argument('--model_type', type=str, default="ann")
    parser.add_argument('--conv_channels', type=list, default=[16, 32])
    parser.add_argument('--fc_dims', type=list, default=[128, 256])
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--padding', type=int, default=1)
    
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--optimizer', type=str, default="Adam")
    parser.add_argument('--scheduler', type=str, default="lambdaLR")
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    
    parser.add_argument('--logging_steps', type=int, default=200)
    parser.add_argument('--save_steps', type=int, default=600)
    
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--predict', action="store_true")
    parser.add_argument('--no_cuda', action="store_true")
    
    args = parser.parse_args()
    
    return args


def main(args):
    device = get_device(args)
    set_seed(args)
    
    # create dir
    if args.train:
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir, exist_ok=True)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
    
    # load and normalize dataset
    def image_transform(image_np):
        output = to_tensor(image_np, normalize=True)
        output = output.reshape(output.size(0), -1) # ANN용 reshaper
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
    model = get_model(args).to(device)
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)
    
    # load model
    if args.ckpt_name is not None:
        ckpt = torch.load(args.ckpt_name)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        steps = ckpt["steps"]
    
    # train
    if args.train:
        trainer = Trainer(
            args,
            train_ds=mnist_train,
            test_ds=mnist_test,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )
        trainer.train()
    
    # test
    if args.test:
        tester = Tester(
            args,
            test_ds=mnist_test,
            model=model,
            optimizer=optimizer,
            device=device
        )
        tester.test()
        
    # predict
    if args.predict:
        predictor = Predictor(
            args,
            model=model,
            optimizer=optimizer,
            device=device
        )
        predictor.predict()
    
    
if __name__ == "__main__":
    
    args = config()
    main(args)