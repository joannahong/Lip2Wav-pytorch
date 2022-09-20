import torch
import torch.nn as nn
from torch.nn import functional as F

class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

class ConvNorm2D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=1, stride=1, padding=None, dilation=1, bias=True, w_init_gain='linear',
                 activation=torch.nn.ReLU, residual=False):
        """
        三维卷积归一化层
        :param in_channels: 2维卷积归一化的输入维度
        :param out_channels: 2维卷积归一化的输出维度
        :param kernel_size, stride, padding, dilation, bias: 控制 nn.Conv2d 的相关参数
        :param w_init_gain: 权值初始化方法
        :param activation: 激活函数
        :param residual: 是否增加残余量
        """
        super(ConvNorm2D, self).__init__()

        if padding is None:
            if type(kernel_size) == tuple:
                ks1, ks2 = kernel_size
            else:
                ks1 = ks2 = kernel_size
            if type(dilation) == tuple:
                dl1, dl2 = dilation
            else:
                dl1 = dl2 = dilation
            assert(ks1 % 2 == 1 and ks2 % 2 == 1)
            padding = (int(dl1 * (ks1 - 1) / 2), int(dl2 * (ks2 - 1) / 2))

        self.conv2d = torch.nn.Conv2d(in_channels, out_channels,
                                      kernel_size=kernel_size, stride=stride, padding=padding,
                                      dilation=dilation, bias=bias)
        # 对 conv2d 进行权值初始化，初始化采用 Xavier 方法
        torch.nn.init.xavier_uniform_(self.conv2d.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

        self.batched = torch.nn.BatchNorm2d(out_channels)
        self.activation = activation()
        self.residual = residual
    def forward(self, signal):
        conv_signal = self.conv2d(signal)

        batched_signal = self.batched(conv_signal)

        if self.residual:
            batched_signal = batched_signal + signal
        activated_signal = self.activation(batched_signal)
        return activated_signal

class ConvNorm3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=1, stride=1, padding=None, dilation=1, bias=True, w_init_gain='linear',
                 activation=torch.nn.ReLU, residual=False):
        super(ConvNorm3D, self).__init__()

        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv3d = torch.nn.Conv3d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride, padding=padding,
                                    dilation=dilation, bias=bias)
        # 对 conv3d 进行权值初始化，初始化采用 Xavier 方法
        torch.nn.init.xavier_uniform_(self.conv3d.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

        self.batched = torch.nn.BatchNorm3d(out_channels)
        self.activation = activation()
        self.residual = residual
    def forward(self, signal):
        conv_signal = self.conv3d(signal)

        batched_signal = self.batched(conv_signal)

        if self.residual:
            batched_signal = batched_signal + signal
        activated_signal = self.activation(batched_signal)

        return activated_signal

class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention