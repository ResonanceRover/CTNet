import os
import math
import torch
import numpy as np
import torch.nn as nn
import scipy.io as sio
from scipy.io import loadmat
from matplotlib import pyplot as plt


class ModuleNotImplementedError(NotImplementedError):
    def __init__(self, message="The requested module type has not been implemented."):
        super().__init__(message)


def create_tfa_module(module_type, **kwargs):
    """Create time-frequency analysis module"""
    if module_type == 'ctn':
        assert kwargs['tfr_size'] == kwargs['feature_dim'] * kwargs['upsampling'], \
            f"tfr_size ({kwargs['tfr_size']}) must be equal to feature_dim ({kwargs['feature_dim']}) * upsampling ({kwargs['upsampling']})"
        return CTNet(red_filters=kwargs['red_filters'], feature_dim=kwargs['feature_dim'], red_layers=kwargs['red_layers'])
    else:
        raise ModuleNotImplementedError(f"Frequency representation module type '{module_type}' not implemented")


def tfa_module(args):
    """Set time-frequency analysis module according to parameters"""
    module = create_tfa_module(args.tfa_module_type, **vars(args))
    if args.use_cuda:
        module.cuda()
    return module
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)

    def forward(self, x):
        avg_pool = self.global_avg_pool(x).view(x.size(0), -1)
        channel_attention = self.fc2(self.relu(self.fc1(avg_pool))).unsqueeze(2).unsqueeze(3)

        return channel_attention

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        combined = torch.cat((max_pool, avg_pool), dim=1)
        spatial_attention = self.conv(combined)

        return spatial_attention

class CBAModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAModule, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        channel_att = self.channel_attention(x)
        spatial_att = self.spatial_attention(x)
        attention = channel_att * spatial_att
        out = x * attention

        return out
class CTNet(nn.Module):
    def __init__(self,  red_filters=16, red_layers=20, feature_dim=200, in_kernel_size=1, out_filters=16, cbam_filters=16
                 ):
        super().__init__()

        self.red_filters = red_filters
        self.feature_dim = feature_dim
        self.red_layers = red_layers
        self.in_kernel_size = in_kernel_size
        self.out_filters = out_filters
        self.cbam_filters = cbam_filters

        self.in_layer=nn.Conv2d(1, red_filters, in_kernel_size,bias=False)
        self.rednet = REDNet(red_layers,num_features=red_filters)
        self.cbam = CBAModule(cbam_filters)
        self.out_layer = nn.ConvTranspose2d(out_filters, 1, (3, 1), stride=(1, 1),
                                            padding=(1, 0), output_padding=(0, 0), bias=False)

    def forward(self, x):
        n = 200
        x = x.reshape((1, 1, n, 2*n))
        s = np.zeros([1, 1, n, 2*n]).astype(np.float32)
        s[:, 0:1, :, :] = (x)
        x = torch.from_numpy(s)
        print('prepare red', x.shape)
        #x=x.cuda()
        x = self.in_layer(x)
        print('in_layer', x.shape)
        x = self.rednet(x)
        x = self.cbam(x)
        x = self.out_layer(x).squeeze(-3)#.transpose(1, 2)
        # plt.figure()
        # plt.ion()
        # plt.imshow(x[0, :, :].abs() / torch.max(x[0, :, :].abs()),extent=[0, 4, 0, fs // 2],  aspect='auto')
        # plt.xticks([])
        # plt.yticks([])
        # plt.gca().invert_yaxis()
        # plt.xlabel('Time')
        # plt.ylabel('Frequency (Hz)')
        # plt.title('Time-Frequency Plot')
        # plt.colorbar(label='Intensity')
        # plt.show()
        return x


class REDNet(nn.Module):
    def __init__(self, num_layers=15, num_features=8):
        super(REDNet, self).__init__()
        self.num_layers = num_layers

        conv_layers = []
        deconv_layers = []

        conv_layers.append(nn.Sequential(nn.Conv2d(num_features, num_features*2, kernel_size=3, stride=2, padding=1),
                                         nn.ReLU(inplace=True)))
        for i in range(num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(num_features*2, num_features*2, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True)))

        for i in range(num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(num_features*2, num_features*2, kernel_size=3, padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.ConvTranspose2d(num_features*2, num_features, kernel_size=3, stride=2, padding=1, output_padding=1))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
                conv_feats.append(x)

        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
                x = x + conv_feat
                x = self.relu(x)

        x += residual
        x = self.relu(x)

        return x