import torch
from torch import nn
import torch.nn.functional as F
import math

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# Probability model
class SoundNet(nn.Module):
    def __init__(self):
        super(SoundNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(64, 1), stride=(2, 1), padding=(32, 0))
        self.batchnorm1 = nn.BatchNorm2d(16, eps=1e-5, momentum=0.1)
        self.relu1 = nn.ReLU(True)
        self.maxpool1 = nn.MaxPool2d((8, 1), stride=(8, 1))

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(32, 1), stride=(2, 1), padding=(16, 0))
        self.batchnorm2 = nn.BatchNorm2d(32, eps=1e-5, momentum=0.1)
        self.relu2 = nn.ReLU(True)
        self.maxpool2 = nn.MaxPool2d((8, 1), stride=(8, 1))

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(16, 1), stride=(2, 1), padding=(8, 0))
        self.batchnorm3 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.relu3 = nn.ReLU(True)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=(8, 1), stride=(2, 1), padding=(4, 0))
        self.batchnorm4 = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1)
        self.relu4 = nn.ReLU(True)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=(4, 1), stride=(2, 1), padding=(2, 0))
        self.batchnorm5 = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1)
        self.relu5 = nn.ReLU(True)
        self.maxpool5 = nn.MaxPool2d((4, 1), stride=(4, 1))

        self.conv6 = nn.Conv2d(256, 512, kernel_size=(4, 1), stride=(2, 1), padding=(2, 0))
        self.batchnorm6 = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1)
        self.relu6 = nn.ReLU(True)

        self.conv7 = nn.Conv2d(512, 1024, kernel_size=(4, 1), stride=(2, 1), padding=(2, 0))
        self.batchnorm7 = nn.BatchNorm2d(1024, eps=1e-5, momentum=0.1)
        self.relu7 = nn.ReLU(True)

        # self.conv8 = nn.Conv2d(1024, 1000, kernel_size=(8, 1), stride=(2, 1))

    def forward(self, waveform):
        x = self.conv1(waveform.unsqueeze(1).permute(0,1,3,2))
        # print(f'conv1: {x.shape}')
        x = self.batchnorm1(x)
        # print(f'batchnorm1: {x.shape}')
        x = self.relu1(x)
        # print(f'relu1: {x.shape}')
        x = self.maxpool1(x)
        # print(f'maxpool1: {x.shape}')

        x = self.conv2(x)
        # print(f'conv2: {x.shape}')
        x = self.batchnorm2(x)
        # print(f'batchnorm2: {x.shape}')
        x = self.relu2(x)
        # print(f'relu2: {x.shape}')
        x = self.maxpool2(x)
        # print(f'maxpool2: {x.shape}')

        x = self.conv3(x)
        # print(f'conv3: {x.shape}')
        x = self.batchnorm3(x)
        # print(f'batchnorm3: {x.shape}')
        x = self.relu3(x)
        # print(f'relu3: {x.shape}')

        x = self.conv4(x)
        # print(f'conv4: {x.shape}')
        x = self.batchnorm4(x)
        # print(f'batchnorm4: {x.shape}')
        x = self.relu4(x)
        # print(f'relu4: {x.shape}')

        x = self.conv5(x)
        # print(f'conv5: {x.shape}')
        x = self.batchnorm5(x)
        # print(f'batchnorm5: {x.shape}')
        x = self.relu5(x)
        # print(f'relu5: {x.shape}')
        # x = self.maxpool5(x)

        x = self.conv6(x)
        # print(f'conv6: {x.shape}')
        x = self.batchnorm6(x)
        # print(f'batchnorm6: {x.shape}')
        x = self.relu6(x)
        # print(f'relu6: {x.shape}')

        x = self.conv7(x)
        # print(f'conv7: {x.shape}')
        x = self.batchnorm7(x)
        # print(f'batchnorm7: {x.shape}')
        x = self.relu7(x)
        # print(f'relu7: {x.shape}')

        # x = self.conv8(x)
        x = x.reshape(x.shape[0],-1)
        # print(x.shape)
        return x

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout=0.1, max_len=3040):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Compute the positional encodings once in log space.
        positional_encoding = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) *
                             -(math.log(10000.0) / dim))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        # positional_encoding = positional_encoding.repeat(16, 1, 1)
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x, combined_mask_tensor):
        batch_size = x.shape[0]
        max_true = x.shape[1]
        masked_positional_encoding = self.positional_encoding.repeat(batch_size, 1, 1)
        masked_positional_encoding = masked_positional_encoding[combined_mask_tensor]
        masked_positional_encoding = torch.reshape(masked_positional_encoding, (batch_size, max_true, 1024))
        # print(x.shape, masked_positional_encoding.shape)
        x = x + masked_positional_encoding
        return self.dropout(x)


class ViT(nn.Module):
    def __init__(self, *, unit_size, dim, depth, heads, mlp_dim, dim_head = 64, dropout = 0., emb_dropout = 0.,max_len):
        super().__init__()

        self.to_unit_embedding = nn.Linear(unit_size, dim)
        self.positional_encoding = PositionalEncoding(dim,max_len=max_len)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, convo, combined_mask_tensor):
        x = self.to_unit_embedding(convo)
        x = self.positional_encoding(x, combined_mask_tensor)
        x = self.transformer(x)
        return x