# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    "FAN"
]


def _chars_to_onehot(chars: torch.LongTensor, dim: int) -> torch.Tensor:
    """Convert characters to one hot encoding

    Args:
        chars (torch.LongTensor): The input of chars
        dim (int): number of string dimensions

    Returns:
        onehot (torch.Tensor): char one hot

    """
    chars = chars.unsqueeze(1)
    onehot = torch.full([chars.size(0), dim], 0.0, dtype=torch.float, device=chars.device)
    onehot = onehot.scatter_(1, chars, 1)

    return onehot


class _ResNetBasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Implement the basic feature extraction block in the residual network

        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution

        """
        super(_ResNetBasicBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=True),
            nn.ReLU(True),
        )
        self.relu = nn.ReLU(True)

        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0), bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.layers(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.add(out, identity)
        out = self.relu(out)

        return out


class _ResNet32(nn.Module):
    """Implement ResNet with 32-layer residual network"""

    def __init__(self):
        super(_ResNet32, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0)),
        )

        self.resnet_block1 = nn.Sequential(
            _ResNetBasicBlock(64, 128),
            _ResNetBasicBlock(128, 128),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0)),
        )

        self.resnet_block2 = nn.Sequential(
            _ResNetBasicBlock(128, 256),
            _ResNetBasicBlock(256, 256),
            _ResNetBasicBlock(256, 256),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
        )

        self.resnet_block3 = nn.Sequential(
            _ResNetBasicBlock(256, 512),
            _ResNetBasicBlock(512, 512),
            _ResNetBasicBlock(512, 512),
            _ResNetBasicBlock(512, 512),
            _ResNetBasicBlock(512, 512),
            _ResNetBasicBlock(512, 512),
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

        self.resnet_block4 = nn.Sequential(
            _ResNetBasicBlock(512, 512),
            _ResNetBasicBlock(512, 512),
            _ResNetBasicBlock(512, 512),
            _ResNetBasicBlock(512, 512),
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, (2, 2), (2, 1), (0, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

        self.conv_block6 = nn.Sequential(
            nn.Conv2d(512, 512, (2, 2), (1, 1), (0, 0), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_block1(x)
        out = self.resnet_block1(out)

        out = self.conv_block2(out)
        out = self.resnet_block2(out)

        out = self.conv_block3(out)
        out = self.resnet_block3(out)

        out = self.conv_block4(out)
        out = self.resnet_block4(out)

        out = self.conv_block5(out)
        out = self.conv_block6(out)

        return out


class _AttentionCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, embedding_size: int) -> None:
        """

        Args:
            input_size (int): The number of input features
            hidden_size (int): The number of hidden features
            embedding_size (int): The number of embedding features

        """
        super(_AttentionCell, self).__init__()
        # Input layer to hidden layer
        self.input_to_hidden = nn.Linear(input_size, hidden_size, bias=False)
        # Hidden layer to hidden layer
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size)
        # Calculate the hidden layer score
        self.hidden_score = nn.Linear(hidden_size, 1, bias=False)
        self.lstm_cell = nn.LSTMCell(input_size + embedding_size, hidden_size)

    def forward(self,
                prev_hidden: torch.Tensor,
                batch_hidden: torch.Tensor,
                chars_onehot: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        """

        Args:
            prev_hidden (torch.Tensor): Prev hidden layer state
            batch_hidden (torch.Tensor): Current hidden layer state
            chars_onehot (torch.Tensor): One hot encoding of prediction results

        Returns:
            hidden_out (torch.Tensor): Current hidden layer output

        """
        # [batch_size x num_encoder_step x input_size] -> [batch_size x num_encoder_step x hidden_size]
        batch_hidden_projection = self.input_to_hidden(batch_hidden)
        # [batch_size x num_encoder_step x hidden_size] -> [batch_size x num_encoder_step x hidden_size, 1]
        prev_hidden_projection = self.hidden_to_hidden(prev_hidden[0]).unsqueeze(1)
        # batch_size x num_encoder_step * 1
        hidden_score = self.hidden_score(torch.tanh(batch_hidden_projection + prev_hidden_projection))
        # Get the weight coefficient of the neuron
        hidden_weights = F.softmax(hidden_score, 1)
        # batch_size x num_channel
        context = torch.bmm(hidden_weights.permute(0, 2, 1), batch_hidden).squeeze(1)
        # batch_size x (num_channel + embedding_size)
        concat_context = torch.cat([context, chars_onehot], 1)
        hidden_out = self.lstm_cell(concat_context, prev_hidden)

        return hidden_out


class _Attention(nn.Module):

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_classes: int) -> None:
        """

        Args:
            input_size (int): The number of input features
            hidden_size (int): The number of hidden features
            num_classes (int): The feature classes number

        """
        super(_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.attention_cell = _AttentionCell(input_size, hidden_size, num_classes)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self,
                batch_hidden: torch.Tensor,
                context: torch.Tensor,
                max_length: int,
                train_mode: bool) -> torch.Tensor:
        batch_size = batch_hidden.size(0)
        # Add [s] token
        context_steps = max_length + 1

        # The number of hidden layer outputs is defined by the number of semantics
        hidden_outputs = torch.full([batch_size, context_steps, self.hidden_size],
                                    0.0,
                                    dtype=torch.float,
                                    device=batch_hidden.device)
        hidden = [
            torch.full([batch_size, self.hidden_size], 0.0, dtype=torch.float, device=batch_hidden.device),
            torch.full([batch_size, self.hidden_size], 0.0, dtype=torch.float, device=batch_hidden.device),
        ]

        if train_mode:
            for context_step in range(context_steps):
                # Convert number(char length) of classes to onehot encoding
                chars_onehot = _chars_to_onehot(context[:, context_step], self.num_classes)
                hidden = self.attention_cell(hidden, batch_hidden, chars_onehot)
                # LSTM hidden index (0: hidden, 1: Cell)
                hidden_outputs[:, context_step, :] = hidden[0]
            probs = self.classifier(hidden_outputs)
        else:
            # Define [GO] token
            targets = torch.full([batch_size], 0.0, dtype=torch.long, device=batch_hidden.device)
            probs = torch.full([batch_size, context_steps, self.num_classes],
                               0.0,
                               dtype=torch.float,
                               device=batch_hidden.device)

            for context_step in range(context_steps):
                # Convert number(char length) of classes to onehot encoding
                chars_onehot = _chars_to_onehot(targets, self.num_classes)
                hidden = self.attention_cell(hidden, batch_hidden, chars_onehot)
                probs_step = self.classifier(hidden[0])
                probs[:, context_step, :] = probs_step
                _, next_targets = probs_step.max(1)
                targets = next_targets

        return probs


class FAN(nn.Module):

    def __init__(self, num_classes: int) -> None:
        super(FAN, self).__init__()
        self.features = _ResNet32()
        self.avgpool = nn.AdaptiveAvgPool2d([None, 1])
        self.attention = _Attention(512, 256, num_classes)

        # Initialize model weights
        self._initialize_weights()

    def forward(self,
                      x: torch.Tensor,
                      context: torch.Tensor,
                      max_length: int,
                      train_mode: bool) -> torch.Tensor:
        return self._forward_impl(x, context, max_length, train_mode)

    # Support torch.script function
    def _forward_impl(self,
                      x: torch.Tensor,
                      context: torch.Tensor,
                      max_length: int,
                      train_mode: bool) -> torch.Tensor:
        # Feature sequence
        features = self.features(x)
        # [b, c, h, w] -> [b, w, c, h]
        features = features.permute([0, 3, 1, 2])
        # Keep the output feature map height at 1
        features = self.avgpool(features)
        # [b, w, c, h] -> [b, w, c]
        features = features.squeeze(3)

        # Attention layer
        out = self.attention(features.contiguous(), context, max_length, train_mode)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
