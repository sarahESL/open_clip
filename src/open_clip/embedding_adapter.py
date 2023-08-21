import torch


class Adapter(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Adapter, self).__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)


    def forward(self, x):
        output = self.linear(x).to(torch.float32)
        return output
