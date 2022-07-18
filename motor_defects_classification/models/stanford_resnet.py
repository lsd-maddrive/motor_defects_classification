import math

import torch
from torch import nn
from torch.nn import functional as F

params = {
    "conv_subsample_lengths": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
    "conv_filter_length": 15,
    "conv_num_filters_start": 32,
    "conv_init": "he_normal",
    "conv_activation": "relu",
    "conv_dropout": 0.2,
    "conv_num_skip": 2,
    "conv_increase_channels_at": 4,
    "ecg_channels": 12,
    "sample_length": 2500,
}


class _bn_relu(nn.Module):
    def __init__(self, in_filters, dropout=0):
        super().__init__()
        self.dropout_rate = dropout

        self.batch_norm = nn.BatchNorm1d(num_features=in_filters)
        self.activation = nn.ReLU(inplace=False)
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout, inplace=False)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.activation(x)
        if self.dropout_rate > 0:
            x = self.dropout(x)
        return x


class add_conv_weight(nn.Module):  # noqa: D101
    def __init__(
        self, in_filters, out_filters, filter_length, subsample_length=1
    ):  # padding for first 2 layers
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_filters,  # 12 for layer1; previous out_channels for others
            out_channels=out_filters,
            kernel_size=filter_length,
            stride=subsample_length,
            padding=filter_length // 2,
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class resnet_block(nn.Module):  # noqa: D101
    def __init__(self, in_filters, out_filters, subsample_length, block_index):
        super().__init__()
        layers = []

        self.shortcut = nn.MaxPool1d(
            kernel_size=subsample_length,
            padding=0,  # Instead of zeropad and adaptive
            dilation=1,  # Default in pytorch, no such param in keras
        )

        for i in range(params["conv_num_skip"]):
            if not (block_index == 0 and i == 0):
                dropout = params["conv_dropout"] if i > 0 else 0
                layer = _bn_relu(dropout=dropout, in_filters=in_filters)
                layers.append(layer)

            if i == 0:
                subsample_len = subsample_length
            else:
                subsample_len = 1

            layer = add_conv_weight(
                in_filters=in_filters,
                out_filters=out_filters,
                filter_length=params["conv_filter_length"],
                subsample_length=subsample_len,
            )
            layers.append(layer)
            in_filters = out_filters
            self.layers = nn.Sequential(*layers)
        # https://github.com/awni/ecg/blob/c97bb96721c128fe5aa26a092c7c33867f283997/ecg/network.py#L63
        self._zero_shortcut_padding = (
            block_index % params["conv_increase_channels_at"] == 0 and block_index > 0
        )

    def forward(self, x):
        x1 = self.shortcut(x)
        x2 = self.layers(x)
        if self._zero_shortcut_padding:
            x1 = torch.cat((x1, torch.zeros_like(x1)), dim=1)
        pad_full = x2.size()[-1] - x1.size()[-1]
        if pad_full > 0:
            pad_left = pad_full // 2
            pad_v = (pad_left, pad_full - pad_left)
            x1 = F.pad(x1, pad_v, "constant", 0)

        x = torch.add(x1, x2)
        return x


def get_num_filters_at_index(index, num_start_filters):
    return 2 ** int(index / params["conv_increase_channels_at"]) * num_start_filters


class add_resnet_layers(nn.Module):  # noqa: D101
    def __init__(self, num_classes):
        super().__init__()
        layers = []

        layer = add_conv_weight(
            in_filters=params["ecg_channels"],
            out_filters=params["conv_num_filters_start"],  # 32
            filter_length=params["conv_filter_length"],
            subsample_length=1,
        )
        layers.append(layer)

        # передать shape
        in_filters = params["conv_num_filters_start"]
        layer = _bn_relu(in_filters=in_filters)
        layers.append(layer)

        for index, subsample_length in enumerate(params["conv_subsample_lengths"]):
            if index > 0:
                in_filters = get_num_filters_at_index(index - 1, params["conv_num_filters_start"])
            out_filters = get_num_filters_at_index(index, params["conv_num_filters_start"])
            layer = resnet_block(in_filters, out_filters, subsample_length, index)
            layers.append(layer)
            in_filters = out_filters

        layer = _bn_relu(in_filters=in_filters)
        layers.append(layer)

        # flatten
        self._flat_layer = torch.nn.Flatten()
        # layers.append(layer)

        flattened_size = params["sample_length"]
        for length in params["conv_subsample_lengths"]:
            if length == 1:
                pass
            else:
                flattened_size = math.ceil(flattened_size / length)

        self._linear_layer = nn.Linear(
            in_features=out_filters * flattened_size, out_features=num_classes
        )
        # layers.append(layer)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        x = self._flat_layer(x)
        x = self._linear_layer(x)
        return x


class StanfordResNet(nn.Module):
    """Class implementation for Stanford ResNet."""

    def __init__(self, **kwargs):
        super().__init__()
        self.layers = add_resnet_layers(num_classes=kwargs.get("num_classes"))

    def forward(self, x):
        x = self.layers(x)
        return x
