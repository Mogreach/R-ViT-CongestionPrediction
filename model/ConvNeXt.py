import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_


class layer_Norm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format='channels_last'):
        super(layer_Norm, self).__init__()
        self.weight =nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ['channels_last', 'channels_first']:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        # [batch_size, height, weight, channel]
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == 'channels_first':
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super(Block, self).__init__()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = layer_Norm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    def __init__(self, in_channels=3, num_class=150, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., layer_scale_init_value=1e-6, head_init_scale=1.):
        super(ConvNeXt, self).__init__()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            layer_Norm(dims[0], eps=1e-6, data_format='channels_first')
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                layer_Norm(dims[i], eps=1e-6, data_format='channels_first'),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.norm = layer_Norm(dims[-1], eps=1e-6, data_format='channels_first')
        self.head = nn.Linear(dims[-1], num_class)

        self.apply(self._init_weights)

        self.x4_12 = nn.Conv2d(768, num_class, kernel_size=1)
        self.norm_x4_12 = layer_Norm(num_class, eps=1e-6, data_format='channels_first')
        self.act = nn.GELU()

        self.deconv_x4 = nn.ConvTranspose2d(num_class, num_class, 4, 2, 1)
        self.norm_x3 = layer_Norm(384, eps=1e-6, data_format='channels_first')
        self.x3_12 = nn.Conv2d(384, num_class, kernel_size=1)

        self.deconv_x3 = nn.ConvTranspose2d(num_class, num_class, 4, 2, 1)

        self.norm_x2 = layer_Norm(192, eps=1e-6, data_format='channels_first')
        self.x2_12 = nn.Conv2d(192, num_class, kernel_size=1)

        self.deconv_x2 = nn.ConvTranspose2d(num_class, num_class, 4, 2, 1)

        self.norm_x1 = layer_Norm(96, eps=1e-6, data_format='channels_first')
        self.x1_12 = nn.Conv2d(96, num_class, kernel_size=1)

        self.upsample = nn.ConvTranspose2d(num_class, num_class, 8, 4, 2, bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(150, 1, 1),
            nn.Sigmoid()
        )

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):

        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        x1 = x

        x = self.downsample_layers[1](x)
        x = self.stages[1](x)
        x2 = x

        x = self.downsample_layers[2](x)
        x = self.stages[2](x)
        x3 = x

        x = self.downsample_layers[3](x)
        x = self.stages[3](x)
        x = self.norm(x)
        return x, x3, x2, x1

    def forward(self, x):
        x, x3, x2, x1 = self.forward_features(x)
        x4_12 = self.act(self.norm_x4_12(self.x4_12(x)))
        x4_x3 = self.act(self.norm_x4_12(self.deconv_x4(x4_12)))

        x3_norm = self.norm_x3(x3)
        x3_12 = self.act(self.norm_x4_12(self.x3_12(x3_norm)))

        x3_x4 = x4_x3 + x3_12

        x43 = self.deconv_x3(x3_x4)

        x2_norm = self.norm_x2(x2)
        x2_12 = self.act(self.norm_x4_12(self.x2_12(x2_norm)))

        x2_x3 = x2_12 + x43

        x32 = self.deconv_x2(x2_x3)

        x1_norm = self.norm_x1(x1)
        x1_12 = self.act(self.norm_x4_12(self.x1_12(x1_norm)))

        x2_x1 = x32 + x1_12

        x = self.upsample(x2_x1)
        x = self.conv(x)
        return x


# rgb = torch.randn([1, 3, 256, 256])
# net = ConvNeXt()
# out = net(rgb)
# print(out.shape)
