"""
model.py
--------
Exact replica of the CNN-LSTM architecture used during training on Kaggle.
This must match the training code 1:1 for load_state_dict() to work.

Architecture:
  Input:  (B, 1, 120, 174) — batch × channel × mfcc_features × time_frames
  Output: (B, 8)           — raw logits for 8 emotion classes
"""

import torch.nn as nn
import torch.nn.functional as F


class CNNLSTM(nn.Module):
    def __init__(self, num_classes: int = 8):
        super(CNNLSTM, self).__init__()

        # CNN Block 1: (B, 1, 120, 174) → (B, 32, 60, 87)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.25)

        # CNN Block 2: (B, 32, 60, 87) → (B, 64, 30, 43)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(0.25)

        # CNN Block 3: (B, 64, 30, 43) → (B, 128, 15, 21)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(0.30)

        # LSTM: width (21) is sequence length, features = 128 * 15 = 1920
        self.lstm  = nn.LSTM(input_size=1920, hidden_size=128, batch_first=True, dropout=0.3)
        self.lstm2 = nn.LSTM(input_size=128,  hidden_size=64,  batch_first=True, dropout=0.3)

        # Classifier head
        self.fc1     = nn.Linear(64, 64)
        self.drop_fc = nn.Dropout(0.40)
        self.fc2     = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.drop1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool3(F.relu(self.bn3(self.conv3(x)))))

        b, c, h, w = x.size()
        x = x.view(b, c * h, w)   # (B, 1920, 21)
        x = x.permute(0, 2, 1)    # (B, 21, 1920)

        x, _ = self.lstm(x)       # (B, 21, 128)
        x, _ = self.lstm2(x)      # (B, 21, 64)
        x = x[:, -1, :]           # (B, 64) — last time step

        x = F.relu(self.fc1(x))
        x = self.drop_fc(x)
        x = self.fc2(x)           # (B, 8) raw logits
        return x
