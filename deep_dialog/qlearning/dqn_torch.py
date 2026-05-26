import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.2):
        super(DQN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.value_fc = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.value_dropout = nn.Dropout(dropout_rate)
        self.advantage_fc = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.advantage_dropout = nn.Dropout(dropout_rate)
        
        self.value_out = nn.Linear(self.hidden_size // 2, 1)
        self.advantage_out = nn.Linear(self.hidden_size // 2, self.output_size)
        
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for layer in [self.fc1, self.value_fc, self.advantage_fc, self.value_out, self.advantage_out]:
            nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain('relu'))
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x, training=True):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x) if training else x
        
        v = F.relu(self.value_fc(x))
        v = self.value_dropout(v) if training else v
        
        a = F.relu(self.advantage_fc(x))
        a = self.advantage_dropout(a) if training else a
        
        v = self.value_out(v)
        a = self.advantage_out(a)
        return v + a - a.mean(1, keepdim=True)

    def predict(self, x):
        with torch.no_grad():
            y = self.forward(x, training=False)
        return torch.argmax(y, 1)

