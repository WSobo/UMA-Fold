import torch
import torch.nn as nn

class MockMock(nn.Module):
    def forward(self, z, *args, **kwargs):
        return z

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.tri_att_start = nn.Linear(10, 10)
        
net = MyNet()
net.tri_att_start = MockMock()
print("Success!")
