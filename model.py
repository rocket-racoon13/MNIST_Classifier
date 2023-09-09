import torch
import torch.nn as nn
import torch.nn.functional as F

from linear import LinearLayer


class MNISTClassifierANN(torch.nn.Module):
    
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.fc_module = nn.ModuleList()
        self.dropout = nn.Dropout(
            p=self.args.dropout_rate
        )
        
        # define fc module
        in_features = self.args.image_height * self.args.image_width
        for fc_dim in self.args.fc_dims:
            fc_layer = LinearLayer(
                in_features=in_features,
                out_features=fc_dim
            )
            self.fc_module.append(fc_layer)
            self.fc_module.append(nn.ReLU())
            self.fc_module.append(self.dropout)
            in_features = fc_dim
            
        out_layer = LinearLayer(
            in_features=self.fc_module[-3].out_features,
            out_features=self.args.num_labels
        )
        self.fc_module.append(out_layer)
        
        
    def forward(self, x):
        for layer in self.fc_module:
            x = layer(x)
        
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
        for (conv_channel, kernel_size, stride, padding) in\
            zip(
                self.args.conv_channels,
                self.args.kernel_size,
                self.args.stride,
                self.args.padding
            ):
            conv_layer = nn.Conv2d(
                in_channels=in_channel,
                out_channels=conv_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
            self.conv_module.append(conv_layer)
            self.conv_module.append(nn.ReLU())
            in_channel = conv_channel
            
        # define fc module
        H_out = int((self.args.image_height + 2 * self.args.padding[-1]\
            - self.args.kernel_size[-1] / self.args.stride[-1] + 1))
        W_out = int((self.args.image_width + 2 * self.args.padding[-1]\
            - self.args.kernel_size[-1] / self.args.stride[-1] + 1))
        in_features = H_out * W_out * self.conv_module[-2].out_channels
        
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
        self.fc_module.append(out_layer)
        
        
    def forward(self, x):
        for layer in self.conv_module:
            x = layer(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        for layer in self.fc_module:
            x = layer(x)
        
        return x