import torch
import torch.nn as nn
import math


class Conformer(nn.Module):
    def __init__(self, emb_size, heads, n_classes, mode):
        super().__init__()
        self.embedding = Embedding(emb_size=emb_size)
        self.multi_head_attention = Encoder(n_layer=6, emb_size=emb_size,
                                            heads=heads, dropout=0.2, hidden_size=64)
        self.classifier = Classifier(n_classes=n_classes, mode=mode)

    def forward(self, x):
        # print(x.size())   # [128, 4800, 2]
        x = torch.transpose(x, 1, 2)
        x = self.embedding(x)   # [b, 314, emb_size]
        x = self.multi_head_attention(x)    # [b, 314, emb_size]
        x = self.classifier(x)
        return x


class Classifier(nn.Module):
    def __init__(self, n_classes, mode=True):
        super(Classifier, self).__init__()
        self.mode = mode
        self.linear = nn.Sequential(
            nn.Linear(314*32, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),

        )
        self.classifier = nn.Linear(1024, n_classes)

    def forward(self, x):
        # [batch_size, length, emb_size]
        # [b, 314, emb_size]
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        if self.mode:
            x = self.classifier(x)
        return x


class Embedding(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.feature_embedding = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 25), stride=(1, 1)),   # (2, 4756)
            nn.Conv2d(32, 32, kernel_size=(2, 1), stride=(1, 1)),   # (1, 4756)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d((1, 75), stride=(1, 15)),      # (1, 314)
            nn.Dropout(0.5),
            nn.Conv2d(32, emb_size, kernel_size=(1, 1), stride=(1, 1))
        )

    def forward(self, x):
        # [b, 2, 4800]
        x = torch.unsqueeze(x, dim=1)   # [b, 1, 2, 4800]
        x = self.feature_embedding(x)   # [b, emb_size, 1, -1]
        # print(x.shape)  # [128, 32, 1, 314]
        x = x.view(x.size(0), x.size(1), -1)    # [b, emb_size, -1] [128, 32, 314]
        x = torch.transpose(x, 1, 2)    # [b, 314, emb_size]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, heads):
        super().__init__()
        self.emb_size = emb_size
        self.heads = heads
        self.w_q = nn.Linear(emb_size, emb_size)
        self.w_k = nn.Linear(emb_size, emb_size)
        self.w_v = nn.Linear(emb_size, emb_size)
        self.w_concat = nn.Linear(emb_size, emb_size)
        self.attention = ScaleDotProductAttention()

    def forward(self, q, k, v):
        q = self.w_q(q)     # [b, -1, emb_size]
        k = self.w_k(k)
        v = self.w_v(v)

        q, k, v = self.split(q), self.split(k), self.split(v)
        out = self.attention(q, k, v)   # [batch_size, heads, length, per_head_size]
        out = self.concat(out)      # [batch_size, length, emb_size]
        out = self.w_concat(out)    # [batch_size, length, emb_size]
        return out

    def split(self, x):
        per_head_size = self.emb_size//self.heads
        x = torch.reshape(x, (x.size(0), x.size(1), self.heads, per_head_size)).transpose(1, 2)
        return x

    def concat(self, x):
        # [batch_size, heads, length, d_tensor]
        x = torch.transpose(x, 1, 2)    # [batch_size, length, heads, d_tensor]
        x = torch.reshape(x, (x.size(0), x.size(1), -1))    # [batch_size, length, emb_size]
        return x


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        # [batch_size, heads, length, per_head_size]
        batch_size, heads, length, per_head_size = k.size()
        k = torch.transpose(k, 2, 3)        # [batch_size, heads, per_head_size, length]
        score = self.softmax((q @ k)/math.sqrt(per_head_size))     # [batch_size, heads, length, length]
        v = score @ v       # [batch_size, heads, length, per_head_size]
        return v


class FeedForward(nn.Module):
    def __init__(self, emb_size, hidden_size, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(emb_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, emb_size)
        )

    def forward(self, x):
        # [batch_size, length, emb_size]
        x = self.linear(x)      # [batch_size, length, emb_size]
        return x


class EncoderLayer(nn.Module):
    def __init__(self, emb_size, heads, hidden_size, dropout):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(emb_size=emb_size, heads=heads)
        self.ffn = FeedForward(emb_size=emb_size, hidden_size=hidden_size, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(emb_size)
        self.layer_norm2 = nn.LayerNorm(emb_size)

    def forward(self, x):
        # [batch_size, length, emb_size]
        _x = x
        x = self.multi_head_attention(x, x, x)  # [batch_size, length, emb_size]
        x = self.layer_norm1(x+_x)
        # [batch_size, length, emb_size]
        _x = x
        x = self.ffn(x)     # [batch_size, length, emb_size]
        x = self.layer_norm2(x+_x)
        return x


class Encoder(nn.Module):
    def __init__(self, n_layer, emb_size, heads, hidden_size, dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(emb_size, heads, hidden_size, dropout) for _ in range(n_layer)]
        )

    def forward(self, x):
        # [batch_size, length, emb_size]
        # [128, 32, 1, 314]
        for layer in self.layers:
            x = layer(x)    # [batch_size, length, emb_size]
        return x

