import torch.nn as nn
import torch.nn.functional as F

class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        # self.fc1 = nn.Linear(in_features=1408, out_features=64) # 9617 167  # 1408 111
        # self.fc2 = nn.Linear(in_features=64, out_features=30)
        # self.output = nn.Linear(in_features=30, out_features=1)

        self.fc1 = nn.Linear(in_features=9617, out_features=100) # 9617 167  # 1408, 111  # 4176
        self.fc2 = nn.Linear(in_features=100, out_features=50)
        self.fc3 = nn.Linear(in_features=50, out_features=25)
        self.fc4 = nn.Linear(in_features=25, out_features=10)
        self.output = nn.Linear(in_features=10, out_features=1)
 
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.softmax(self.output(x))
        return x