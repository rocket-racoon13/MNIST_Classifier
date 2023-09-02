import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils import LinearLayer


class MNISTClassifierANN(torch.nn.Module):
    
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.fc_module = nn.ModuleList()
        
        # define fc module
        in_features = self.args.image_height * self.args.image_width
        for fc_dim in self.args.fc_dims:
            fc_layer = nn.Linear(
                in_features=in_features,
                out_features=fc_dim
            )
            self.fc_module.append(fc_layer)
            self.fc_module.append(nn.ReLU())
            in_features = fc_dim
            
        out_layer = nn.Linear(
            in_features=self.fc_module[-2].out_features,
            out_features=self.args.num_labels
        )
        
        
    def forward(self, x):
        x = self.fc_module(x)
        
        return x
        
        
class MNISTClassifierCNN(torch.nn.Module):
    
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.conv_module = nn.ModuleList()
        self.fc_module = nn.ModuleList()
        self.dropout = nn.Dropout(
            p=self.args.dropout_rate
        )
        
        # define convolution module
        in_channel = self.args.image_channel
        for conv_channel in self.args.conv_channels:
            conv_layer = nn.Conv2d(
                in_channels=in_channel,
                out_channels=conv_channel,
                kernel_size=self.args.kernel_size,
                stride=self.args.stride,
                padding=self.args.padding
            )
            self.conv_module.append(conv_layer)
            self.conv_module.append(nn.ReLU())
            in_channel = conv_channel
            
        # define fc module
        conv_in_size = self.conv_module[-2].in_channels
        conv_out_size = int((conv_in_size + 2 * self.args.padding\
            - self.args.kernel_size) / self.args.stride + 1)
        in_features = conv_out_size * conv_out_size * self.conv_module[-2].out_channels
        
        for fc_dim in self.args.fc_dims:
            fc_layer = nn.Linear(
                in_features=in_features,
                out_features=fc_dim
            )
            self.fc_module.append(fc_layer)
            self.fc_module.append(nn.ReLU())
            in_features = fc_dim
            
        out_layer = nn.Linear(
            in_features=self.fc_module[-2].out_feautres,
            out_features=self.args.num_labels
        )
        self.fc_module.append(out_layer)
        
        
    def forward(self, x):
        x = self.conv_module(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc_module(x)
        
        return x