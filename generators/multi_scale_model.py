import functools

import torch
import torch.nn as nn 
import torch.nn.functional as F
 
class Generator(nn.Module):
    def __init__(self, base_channel, num_layers):
        super(Generator, self).__init__()
        data_dim = 1
        nonlinearity = nn.LeakyReLU(0.1)
        self.encoder = Encoder(data_dim, base_channel, num_layers, nonlinearity)
        self.decoder = MultiScaleDecoder(base_channel, num_layers, nonlinearity)

    def forward(self, x):
        out = self.decoder(self.encoder(x))
        return out
    
    def multi_scale_output(self, x):
        features = self.encoder(x)
        multi_scale_output = self.decoder.multi_scale_output(features)
        return multi_scale_output

class Encoder(nn.Module):
    def __init__(self, data_dim, base_channel, num_layers, 
                 nonlinearity, norm_type='batch_norm', 
                 max_channel=1024):
        super(Encoder, self).__init__()
        assert norm_type in ['instance_norm', 'batch_norm']
        if norm_type == 'instance_norm':
            norm_layer = functools.partial(nn.InstanceNorm3d, affine=True)
        elif norm_type == 'batch_norm':
            norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
 
        self.convs = nn.ModuleList()
        self.convs.append(
            nn.Conv3d(data_dim, base_channel, 3, padding=1), 
        )

        for i in range(num_layers):
            in_channels = min(base_channel * (2 ** i), max_channel)
            out_channels = min(base_channel * (2 ** (i+1)), max_channel)

            self.convs.append(nn.Sequential(
                nonlinearity,
                norm_layer(in_channels),                
                nn.Conv3d(in_channels, out_channels, 3, stride=2, padding=1), 

                nonlinearity,
                norm_layer(out_channels),                
                nn.Conv3d(out_channels, out_channels, 3, padding=1),
            ))
        self.out_channels = out_channels

    def forward(self, x):
        result = []
        for conv in self.convs:
            x = conv(x)
            result.append(x)
        return result[::-1]
        

class MultiScaleDecoder(nn.Module):
    def __init__(self, base_channel, num_layers, nonlinearity, norm_type='batch_norm', dropout_ratio=0.5):
        super(MultiScaleDecoder, self).__init__()
        assert norm_type in ['instance_norm', 'batch_norm']
        if norm_type == 'instance_norm':
            norm_layer = functools.partial(nn.InstanceNorm3d, affine=True)
        elif norm_type == 'batch_norm':
            norm_layer = functools.partial(nn.BatchNorm3d, affine=True)

        self.dropout_ratio = dropout_ratio
        
        in_channels = base_channel * (2 ** num_layers) 
        out_channels = base_channel * (2 ** (num_layers-1))

        self.input = nn.Sequential(
            nonlinearity,
            norm_layer(in_channels),
            nn.ConvTranspose3d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),

            nonlinearity,
            norm_layer(out_channels),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
        )     

        self.convs = nn.ModuleList()
        self.to_outputs = nn.ModuleList()

        for i in range(num_layers-1)[::-1]:             
            in_channels = base_channel * (2 ** (i+1)) * 2
            out_channels = base_channel * (2 ** i)

            self.convs.append(nn.Sequential(
                nonlinearity,
                norm_layer(in_channels),
                nn.ConvTranspose3d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),

                nonlinearity,
                norm_layer(out_channels),                
                nn.Conv3d(out_channels, out_channels, 3, padding=1),                
            ))

            self.to_outputs.append(ToOutput(out_channels, nonlinearity))

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(self.dropout_ratio)

    def forward(self, features):
        x = self.input(features[0])
        skip, index = None, 0
        for conv, to_output in zip(self.convs, self.to_outputs):
            index += 1
            x = torch.cat([x, self.dropout(features[index])], 1)
            x = conv(x)
            skip = to_output(x, skip)
        out = self.sigmoid(skip)
        return out

    def multi_scale_output(self, features):
        x = self.input(features[0])
        skip, index = None, 0
        multi_scale_output = []
        for conv, to_output in zip(self.convs, self.to_outputs):
            index += 1
            x = torch.cat([x, self.dropout(features[index])], 1)
            x = conv(x)
            skip = to_output(x, skip)
            multi_scale_output.append(self.sigmoid(skip))
        out = self.sigmoid(skip)
        multi_scale_output.append(out)
        return multi_scale_output


class ToOutput(nn.Module):
    def __init__(self, input_channel, nonlinearity, ):
        super(ToOutput, self).__init__()
        self.nonlinearity = nonlinearity
        self.conv = nn.Conv3d(input_channel, 1, 3, padding=1)
        
    def forward(self, x, skip=None):
        out = self.nonlinearity(x)
        out = self.conv(out)
        if skip is not None:
            skip = F.interpolate(skip, scale_factor=2)
            out = out + skip
        return out



