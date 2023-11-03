import torch
import torch.nn as nn

from .nn import (
    linear,
    timestep_embedding,
)

class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linearQ= nn.Linear(dim, dim)
        self.linearK= nn.Linear(dim, dim)
        self.linearV= nn.Linear(dim, dim)

    def forward(self, x):
        q = self.linearQ(x).unsqueeze(-1)
        k = self.linearK(x).unsqueeze(-1)
        v = self.linearV(x).unsqueeze(-1)

        attn = torch.matmul(q, k.transpose(-1,-2))
        attn = attn / torch.sqrt(torch.tensor(x.shape[-1]))
        attn = torch.softmax(attn, dim=-1)

        output = torch.matmul(attn, v)

        return output.squeeze(-1)


class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_num=[2000,1000,500,500], patch_size=1000, num_classes=None):
        super().__init__()
        self.patch_size = patch_size
        self.input_dim = input_dim
        self.hidden_num = hidden_num

        self.time_embed = nn.Sequential(
            linear(hidden_num[0], hidden_num[0]),
            nn.SiLU(),
            linear(hidden_num[0], hidden_num[0]),
        )

        self.fc0 = nn.Linear(input_dim, hidden_num[0], bias=True)
        self.emb_layers0 = nn.Sequential(
            nn.SiLU(),
            linear(
                hidden_num[0],
                hidden_num[0],
            ),
        )
        self.norm0 = nn.LayerNorm(hidden_num[0])

        # 前向全连接层和时间
        self.fc1 = nn.Linear(hidden_num[0], hidden_num[1], bias=True)
        self.emb_layers1 = nn.Sequential(
            nn.SiLU(),
            linear(
                hidden_num[0],
                hidden_num[1],
            ),
        )
        self.norm1 = nn.LayerNorm(hidden_num[1])

        self.fc2 = nn.Linear(hidden_num[1], hidden_num[2], bias=True)
        self.emb_layers2 = nn.Sequential(
            nn.SiLU(),
            linear(
                hidden_num[0],
                hidden_num[2],
            ),
        )
        self.norm2 = nn.LayerNorm(hidden_num[2])

        self.fcb1 = nn.Linear(hidden_num[2], hidden_num[1], bias=True)
        self.emb_layersb1 = nn.Sequential(
            nn.SiLU(),
            linear(
                hidden_num[0],
                hidden_num[1],
            ),
        )

        self.fcb2 = nn.Linear(hidden_num[1], hidden_num[0], bias=True)
        self.emb_layersb2 = nn.Sequential(
            nn.SiLU(),
            linear(
                hidden_num[0],
                hidden_num[0],
            ),
        )
        
        self.act = torch.nn.SiLU()
        self.drop = nn.Dropout(0.1)
        
        self.out0 = nn.Linear(hidden_num[0], hidden_num[1]*2, bias=True)
        self.norm00 = nn.LayerNorm(hidden_num[1]*2)
        self.out = nn.Linear(hidden_num[1]*2, input_dim, bias=True)


    def forward(self, x_input, t, y=None):

        emb = self.time_embed(timestep_embedding(t, self.hidden_num[0]).squeeze(1))

        his = []
        x = self.fc0(x_input.float())    # 3000
        x = x+self.emb_layers0(emb)
        x = self.norm0(x)
        x = self.act(x)
        his.append(x)

        x = self.fc1(x)                  # 2000
        x = x+self.emb_layers1(emb)
        x = self.norm1(x)
        x = self.act(x)
        his.append(x)
        
        x = self.fc2(x)                  # 1000
        x = x+self.emb_layers2(emb)
        x = self.norm2(x)
        x = self.act(x)

        x = self.fcb1(x)                 # 2000
        x = x+self.emb_layersb1(emb)
        x = self.norm1(x)
        x = self.act(x)      

        x = x + his.pop()
        x = self.fcb2(x)                 # 3000
        x = x+self.emb_layersb2(emb)
        x = self.norm0(x)
        x = self.act(x)       

        x = x + his.pop()
        x = self.out0(x)
        x = self.norm00(x)
        x = self.act(x)

        x = self.out(x)

        return x


class HALF_MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_num=[2000,1000,500,200], num_class=11):
        super().__init__()
        self.num_class = num_class
        self.input_dim = input_dim
        self.hidden_num = hidden_num

        self.time_embed = nn.Sequential(
            linear(hidden_num[0], hidden_num[1]),
            nn.SiLU(),
            linear(hidden_num[1], hidden_num[1]),
        )

        self.fc = nn.Linear(input_dim, hidden_num[1], bias=True)
        self.emb_layers0 = nn.Sequential(
            nn.SiLU(),
            linear(
                hidden_num[1],
                hidden_num[1],
            ),
        )
        self.norm = nn.LayerNorm(hidden_num[1])
        
        self.fc1 = nn.Linear(hidden_num[1], hidden_num[2], bias=True)
        self.emb_layers1 = nn.Sequential(
            nn.SiLU(),
            linear(
                hidden_num[1],
                hidden_num[2],
            ),
        )
        self.norm1 = nn.LayerNorm(hidden_num[2])

        self.fc2 = nn.Linear(hidden_num[2], hidden_num[3], bias=True)
        self.emb_layers2 = nn.Sequential(
            nn.SiLU(),
            linear(
                hidden_num[1],
                hidden_num[3],
            ),
        )
        self.norm2 = nn.LayerNorm(hidden_num[3])

        self.act = torch.nn.SiLU()
        self.drop = nn.Dropout(0.1)
        self.out = nn.Linear(hidden_num[3], num_class, bias=True)


    def forward(self, x_input, t):
        emb = self.time_embed(timestep_embedding(t, self.hidden_num[0]).squeeze(1))

        x = self.fc(x_input)
        x = x+self.emb_layers0(emb)
        x = self.norm(x)
        x = self.act(x)

        x = self.fc1(x)                  
        x = x+self.emb_layers1(emb)
        x = self.norm1(x)
        x = self.act(x)

        x = self.fc2(x)                  
        x = x+self.emb_layers2(emb)
        x = self.norm2(x)
        x = self.act(x)

        x = self.out(x)

        return x
