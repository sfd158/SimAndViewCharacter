"""
Get code from https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-
"""
   
import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


class Swish(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T: int, d_model: int, dim: int):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t: torch.Tensor):
        """
        Input shape: (*)
        Output shape: (*, dim)
        """
        return self.timembedding(t.long())


class ResBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, tdim: int, dropout: float):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.block1 = nn.Sequential(
            Swish(),
            nn.Conv1d(in_channel, out_channel, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_channel),
        )
        self.block2 = nn.Sequential(
            Swish(),
            nn.Dropout(float(dropout)),
            nn.Conv1d(out_channel, out_channel, 3, stride=1, padding=1),
        )

        self.shortcut = nn.Conv1d(in_channel, out_channel, 1, stride=1, padding=0) if self.in_channel != self.out_channel else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor):
        """
        x.shape == (batch, channel, length)
        temb.shape == (batch, emb dim)
        """
        h = self.block1(x)
        h += self.temb_proj(temb)[..., None]
        h = self.block2(h)
        h = h + self.shortcut(x)
        return h


class SimpleConv1D(nn.Module):

    def __init__(self, T: int, in_channel: int, out_channel: int, hidden_dim: int = 512, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.time_embedding = TimeEmbedding(T, hidden_dim * 2, hidden_dim)
        self.head = nn.Conv1d(in_channel, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.net = nn.ModuleList()
        for i in range(num_layers):
            self.net.append(ResBlock(hidden_dim, hidden_dim, hidden_dim, dropout))
        self.tail = nn.Sequential(
            Swish(),
            nn.Conv1d(hidden_dim, out_channel, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor = None):
        """
        x.shape == (batch, channel, length)
        t.shape == (batch, )
        """
        temb = self.time_embedding(t)
        x = self.head(x)
        for net in self.net:
            x = net(x, temb)
        x = self.tail(x)
        return x

def test_func():
    # time_emb = TimeEmbedding(5000, 32, 64)(torch.ones([512,], dtype=torch.long))
    data = torch.randn([512, 64, 10])
    # net = torch.jit.script(ResBlock(64, 128, 64, 0))
    net = torch.jit.script(SimpleConv1D(5000, 64, 128, 256))(data, torch.ones([512,], dtype=torch.long), None)

if __name__ == "__main__":
    test_func()