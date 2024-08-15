# Copyright 2022 CircuitNet. All rights reserved.

import torch
import torch.nn as nn

from collections import OrderedDict

def generation_init_weights(module):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                    or classname.find('Linear') != -1):
            
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    module.apply(init_func)

def load_state_dict(module, state_dict, strict=False, logger=None):
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None

    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    if len(err_msg) > 0:
        err_msg.insert(
            0, 'The gan_model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)
    return missing_keys

class conv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True):
        super(conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)

class upconv(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(upconv, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(dim_in, dim_out, 4, 2, 1),
                nn.BatchNorm2d(dim_out),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)
class AttentionBlock(nn.Module):
    #x为skip，g为上一层
    def __init__(self, in_channels_x, in_channels_g, int_channels):
        super(AttentionBlock, self).__init__()
        self.Wx = nn.Sequential(nn.Conv2d(in_channels_x, int_channels, kernel_size=1),
                                nn.BatchNorm2d(int_channels))
        self.Wg = nn.Sequential(nn.Conv2d(in_channels_g, int_channels, kernel_size=1),
                                nn.BatchNorm2d(int_channels))
        self.psi = nn.Sequential(nn.Conv2d(int_channels, 1, kernel_size=1),
                                 nn.BatchNorm2d(1),
                                 nn.Sigmoid())

    def forward(self, x, g):
        input = g
        # apply the Wx to the skip connection
        x1 = self.Wx(x)
        # after applying Wg to the input, upsample to the size of the skip connection
        # g1 = nn.functional.interpolate(self.Wg(g), x1.shape[2:], mode='bilinear', align_corners=False)
        g = self.Wg(g)
        resampler = self.psi(nn.ReLU()(x1 + g))
        resampler = nn.Sigmoid()(resampler)
        x_attention = resampler * x
        out = torch.cat((x_attention, input), dim=1)  # 完成拼接
        return out
class AttentionBlock(nn.Module):

    #x为skip，g为上一层
    def __init__(self, in_channels_x, in_channels_g, int_channels):
        super(AttentionBlock, self).__init__()
        self.Wx = nn.Sequential(nn.Conv2d(in_channels_x, int_channels, kernel_size=1),
                                nn.BatchNorm2d(int_channels))
        self.Wg = nn.Sequential(nn.Conv2d(in_channels_g, int_channels, kernel_size=1),
                                nn.BatchNorm2d(int_channels))
        self.psi = nn.Sequential(nn.Conv2d(int_channels, 1, kernel_size=1),
                                 nn.BatchNorm2d(1),
                                 nn.Sigmoid())

    def forward(self, x, g):
        input = g
        # apply the Wx to the skip connection
        x1 = self.Wx(x)
        # after applying Wg to the input, upsample to the size of the skip connection
        # g1 = nn.functional.interpolate(self.Wg(g), x1.shape[2:], mode='bilinear', align_corners=False)
        g = self.Wg(g)
        resampler = self.psi(nn.ReLU()(x1 + g))
        resampler = nn.Sigmoid()(resampler)
        x_attention = resampler * x
        out = torch.cat((x_attention, input), dim=1)  # 完成拼接
        return out

class Encoder(nn.Module):
    def __init__(self, in_dim=3, out_dim=32):
        super(Encoder, self).__init__()
        #input -> E1
        self.in_dim = in_dim
        self.c1 = conv(in_dim, 32)
        #E1 -> E2
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)  # 这里先用这个
        #E2 -> E3
        self.c2 = conv(32, 64)
        #E3 -> E4
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)  # 这里先用这个
        #E4 -> E5
        self.c3 = nn.Sequential(
                nn.Conv2d(64, out_dim, 3, 1, 1),
                nn.BatchNorm2d(out_dim),
                nn.Tanh()
                )

    def init_weights(self):
        """Initialize the weights."""
        generation_init_weights(self)

    def forward(self, input):
        e1 = self.c1(input)
        e2 = self.pool1(e1)
        e3 = self.c2(e2)
        e4 = self.pool2(e3)
        e5 = self.c3(e4)
        return e5,e2,e1


class Decoder(nn.Module):
    def __init__(self, out_dim=2, in_dim=32):
        super(Decoder, self).__init__()
        #input -> D1
        self.conv1 = conv(in_dim, 32)
        #D1 -> D2
        self.upc1 = upconv(32, 16)
        #D2 -> D3
        self.conv2 = conv(16, 16)
        self.attblock1 = AttentionBlock(32,16,int((32+16)/2))
        #D3 -> D4
        self.upc2 = upconv(32+16, 4)
        #D4 -> D5
        self.attblock2 = AttentionBlock(32,4,int((32+4)/2))
        self.conv3 =  nn.Sequential(
                nn.Conv2d(32+4, out_dim, 3, 1, 1),
                nn.Sigmoid()
                )

    def init_weights(self):
        generation_init_weights(self)


    def forward(self, input ,e2,e1):
        skip1=e2
        skip2=e1
        feature= input
        d1 = self.conv1(feature)
        d2 = self.upc1(d1)
        d3 = self.conv2(d2)
        d4 = self.attblock1(skip1,d3)
        d5 = self.upc2(d4)
        d6 = self.attblock2(skip2,d5)
        output = self.conv3(d6)  # shortpath from 2->7
        return output



class Att_RouteNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=1,
                 **kwargs):
        super().__init__()
        self.encoder = Encoder(in_dim=in_channels)
        self.decoder = Decoder(out_dim=out_channels)

    def forward(self, x):
        e5,e2,e1= self.encoder(x)
        output = self.decoder(e5,e2,e1)
        return output

    def init_weights(self, pretrained=None, strict=True, **kwargs):
        if isinstance(pretrained, str):
            new_dict = OrderedDict()
            weight = torch.load(pretrained, map_location='cpu')['state_dict']
            for k in weight.keys():
                new_dict[k] = weight[k]
            load_state_dict(self, new_dict, strict=strict, logger=None)
        elif pretrained is None:
            self.encoder.init_weights()
            self.decoder.init_weights()
        else:
            raise TypeError("'pretrained' must be a str or None. ")

