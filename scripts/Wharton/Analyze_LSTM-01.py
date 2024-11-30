__author__ = 'Yunsong Xie'

import pandas as pd
import torch
from torch import nn


n_layers = 3
n_features = 10
n_hidden_size = 15
model = nn.LSTM(input_size=n_features, hidden_size=n_hidden_size, num_layers=n_layers, dropout=0.15, batch_first=True,
               proj_size=5)

batch = 17
time_length = 7

input_matrix = torch.randn(batch, time_length, n_features)
h0_matrix = torch.randn(n_layers, batch, n_hidden_size)
c0_matrix = torch.randn(n_layers, batch, n_hidden_size)
output, (hn, cn) = model(input_matrix)
output.shape

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

