from diffusers import UNet2DModel
import torch
import torch.nn as nn

class SinusoidalEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_T):
        assert embedding_dim % 2 == 0
        super(SinusoidalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_T = max_T
        # create a lookup table
        embedding_table = torch.zeros((max_T+1, embedding_dim))
        arr_i = torch.arange(embedding_dim//2)
        arr_1000_2i_d = torch.pow(10000, (2*arr_i / embedding_dim))
        for t in range(max_T+1):
            embedding_table[t, 0::2] = torch.sin(t/arr_1000_2i_d)
            embedding_table[t, 1::2] = torch.cos(t/arr_1000_2i_d)

        self.register_buffer("embedding_table", embedding_table)


    def forward(self, timesteps: torch.Tensor):
        assert torch.all(timesteps <= self.max_T)
        # input: batched timestep of dimension [B], each t <= max_T
        # output: sinusoidal embeddings of dimension [B, self.embedding_dim]
        return self.embedding_table[timesteps]

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        _, _, H, W = x.shape
        x = x.view(-1, self.channels, H*W).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, H, W)


class ResBlock_TSEmb(nn.Module):
    def __init__(self, in_channel, out_channel, groups = 32, dropout_rate=0.1, max_T=1000):
        assert out_channel % groups == 0 and in_channel % groups == 0
        super(ResBlock_TSEmb, self).__init__()
        self.gn1 = nn.GroupNorm(groups, in_channel)
        self.nl1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 
                              kernel_size=3,
                              padding=1, 
                              bias=False)

        self.ts_emb = SinusoidalEmbedding(out_channel, max_T)

        self.gn2 = nn.GroupNorm(groups, out_channel)
        self.nl2 = nn.SiLU()
        self.drop = nn.Dropout2d(dropout_rate)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 
                              kernel_size=3,
                              padding=1, 
                              bias=False)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
            nn.GroupNorm(groups, out_channel)
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor):
        out = self.conv1(self.nl1(self.gn1(x)))
        # print(out.shape, self.ts_emb(timesteps)[:, :, None, None].shape)
        out = out + self.ts_emb(timesteps)[:, :, None, None]
        out = self.conv2(self.drop(self.nl2(self.gn2(out))))
        out = out + self.shortcut(x)
        return out

class SwapAxis(nn.Module):
    def __init__(self, axis1, axis2):
        super(SwapAxis, self).__init__()
        self.axis1 = axis1
        self.axis2 = axis2
        
    def forward(self, x: torch.Tensor):
        return x.transpose(self.axis1, self.axis2)

class ResAttentionBlock_TSEmb(ResBlock_TSEmb):
    def __init__(self, in_channel, out_channel, groups = 32, dropout_rate=0.1, max_T=1000, head_count=4):
        super(ResAttentionBlock_TSEmb, self).__init__(in_channel, out_channel, groups, dropout_rate, max_T)
        self.atten = SelfAttention(out_channel)#nn.MultiheadAttention(out_channel, head_count, dropout_rate, batch_first=True)
        
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor):
        # first convolution block
        out = self.conv1(self.nl1(self.gn1(x))) 
        out = out + self.ts_emb(timesteps)[:, :, None, None]
        # Attention        
        out_att = self.atten(out)
        out = out + out_att
        # Second convblock
        out = self.conv2(self.drop(self.nl2(self.gn2(out))))
        out = out + self.shortcut(x)
        return out
        
class DownBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_channel, out_channel, 
                                kernel_size=3, 
                                padding=1, 
                                stride=2, 
                                bias=False)

    def forward(self, x):
        return self.conv2d(x)

class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpBlock, self).__init__()
        self.upsamp = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv2d = nn.Conv2d(in_channel, out_channel, 
                                kernel_size=3, 
                                padding=1, 
                                bias=False)

    def forward(self, x):
        return self.conv2d(self.upsamp(x))

class CustomBasicUNet2DModelGrey(nn.Module):
    def __init__(self, dims, max_T=1000):
        super(CustomBasicUNet2DModelGrey, self).__init__()
        assert dims % 4 == 0
        self.conv_in = nn.Conv2d(1, 32, 
                                kernel_size=3, 
                                padding=1, 
                                bias=False)
        
        self.down1 = nn.ModuleList([ # 28x28 -> 14x14
            ResBlock_TSEmb(32, 32, max_T=max_T),
            ResBlock_TSEmb(32, 32, max_T=max_T),
            DownBlock(32, 32)
        ])
        self.down2 = nn.ModuleList([ # 14x14 -> 7x7
            ResAttentionBlock_TSEmb(32, 64, max_T=max_T),
            ResAttentionBlock_TSEmb(64, 64, max_T=max_T),
            DownBlock(64, 64)
        ])
        self.down3 = nn.ModuleList([ # 7x7 -> 7x7
            ResBlock_TSEmb(64, 64, max_T=max_T),
            ResBlock_TSEmb(64, 64, max_T=max_T)
        ])
        
        self.mid = nn.ModuleList([ # 7x7 -> 7x7
            ResAttentionBlock_TSEmb(64, 64, max_T=max_T),
            ResAttentionBlock_TSEmb(64, 64, max_T=max_T)
        ])
        
        self.up1 = nn.ModuleList([ # 7x7 -> 14x14
            ResBlock_TSEmb(128, 64, max_T=max_T),
            ResBlock_TSEmb(128, 64, max_T=max_T),
            ResBlock_TSEmb(128, 64, max_T=max_T),
            UpBlock(64, 64)            
        ])
        self.up2 = nn.ModuleList([ # 14x14 -> 28x28
            ResAttentionBlock_TSEmb(128, 64, max_T=max_T),
            ResAttentionBlock_TSEmb(128, 64, max_T=max_T),
            ResAttentionBlock_TSEmb(96, 32, max_T=max_T),
            UpBlock(32, 32)            
        ])
        self.up3 = nn.ModuleList([ # 28x28 -> 28x28
            ResBlock_TSEmb(64, 32, max_T=max_T),
            ResBlock_TSEmb(64, 32, max_T=max_T),
            ResBlock_TSEmb(64, 32, max_T=max_T),
        ])
        
        self.conv_out = nn.Conv2d(32, 1, 
                                kernel_size=3, 
                                padding=1, 
                                bias=False)
    
    @staticmethod
    def _block_down_forward(block: list, x: torch.Tensor, timesteps: torch.Tensor):
        outs = []
        out = x
        for b in block:
            # print(b)
            if isinstance(b, ResBlock_TSEmb):
                out = b(out, timesteps)
            else:
                out = b(out)
            outs.append(out)
        return out, outs
    
    @staticmethod        
    def _block_up_forward(block: list, skips: list, x: torch.Tensor, timesteps: torch.Tensor):
        # (B, C, H, W)
        out = x
        for i, b in enumerate(block):
            if isinstance(b, ResBlock_TSEmb):
                # pop a skip conenction out
                skip = skips.pop()
                out = torch.cat((out, skip), 1)
                # print(f'{i}th shape {out.shape}')
                out = b(out, timesteps)
            else:
                out = b(out)
        return out
    
    def forward(self, x, t):
        out = self.conv_in(x)
        skip_conns = [out]
        
        out, skip = self._block_down_forward(self.down1, out, t)
        skip_conns += skip
        out, skip = self._block_down_forward(self.down2, out, t)
        skip_conns += skip
        out, skip = self._block_down_forward(self.down3, out, t)
        skip_conns += skip
        
        out, _ = self._block_down_forward(self.mid, out, t)
        
        # for s in skip_conns:
        #     print(s.shape)
        
        out = self._block_up_forward(self.up1, skip_conns, out, t)
        out = self._block_up_forward(self.up2, skip_conns, out, t)
        out = self._block_up_forward(self.up3, skip_conns, out, t)
        
        out = self.conv_out(out)
        
        return out

class SimpleUNet2DModelGrey(UNet2DModel):
    def __init__(self, dims, num_class):
        super().__init__(
            sample_size=dims, 
            in_channels=1, 
            out_channels=1, 
            dropout=0.1, 
            num_class_embeds=num_class, 
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
            block_out_channels=(32, 64, 64),
        )

class SimpleUNet2DModelRGB(UNet2DModel):
    def __init__(self, dims, num_class):
        super().__init__(
            sample_size=dims, 
            in_channels=3, 
            out_channels=3, 
            dropout=0.1, 
            num_class_embeds=num_class, 
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
            block_out_channels=(96, 192, 192),
        )

class OpenAIUNet2DModelRGB(UNet2DModel):
    def __init__(self, dims, num_class):
        super().__init__(
            sample_size=dims, 
            in_channels=3, 
            out_channels=3, 
            dropout=0.3, 
            num_class_embeds=num_class, 
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D"
            ),
            block_out_channels=(64, 128, 128, 128),
        )

if __name__=='__main__':
    # import matplotlib.pyplot as plt
    # test_se = SinusoidalEmbedding(512, 1000)
    # print(test_se.embedding_table)
    # plt.imshow(test_se.embedding_table, 'viridis')
    # plt.axis('off') # Optional: hides axes
    # plt.savefig('image.png')

    mnistModel = CustomBasicUNet2DModelGrey(28)
    batch = 8
    input_image = torch.randn(batch, 1, 28, 28)
    input_ts = torch.randint(1, 1000, (batch,))
    
    output = mnistModel(input_image, input_ts)
    
    print(output.shape)