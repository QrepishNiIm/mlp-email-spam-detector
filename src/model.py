import torch
import torch.nn as nn

class SpamNet(nn.Module):
    def __init__(self, input_size=57, hidden1=128, hidden2=64, num_classes=2):
        super(SpamNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)

        # ВАЖНО: нет активации на выходе!
        # CrossEntropyLoss сам применяет log_softmax внутри
        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.fc3(x)  # Возвращаем логиты
        return x