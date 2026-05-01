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


    def forward(self, timesteps):
        assert torch.is_tensor(timesteps) and torch.all(timesteps <= self.max_T)
        # input: batched timestep of dimension [B], each t <= max_T
        # output: sinusoidal embeddings of dimension [B, self.embedding_dim]
        return self.embedding_table[timesteps]

class ResBlock_TSEmb(nn.Module):
    def __init__(self, in_channel, out_channel, groups = 32, dropout_rate=0.1):
        assert out_channel % groups == 0 and in_channel % groups == 0
        super(ResBlock_TSEmb, self).__init__()
        self.gn1 = nn.GroupNorm(groups, in_channel)
        self.nl1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 
                              kernel_size=3,
                              padding=1, 
                              bias=False)

        self.shortcut = nn.Sequential()

        self.gn2 = nn.GroupNorm(groups, in_channel)
        self.nl2 = nn.SiLU()
        self.drop = nn.Dropout2d(dropout_rate)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 
                              kernel_size=3,
                              padding=1, 
                              bias=False)

    def forward(self, x, timestep_emb):
        out = self.conv1(self.nl1(self.gn1(x)))
        out += timestep_emb
        out = self.conv2(self.drop(self.nl2(self.gn2(x))))
        out += self.shortcut(x)
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
        super(DownBlock, self).__init__()
        self.upsamp = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv2d = nn.Conv2d(in_channel, out_channel, 
                                kernel_size=3, 
                                padding=1, 
                                bias=False)

    def forward(self, x):
        return self.conv2d(self.upsamp(x))

class CustomUNet2DModelGrey(nn.Module):
    def __init__(self, dims):
        super(CustomUNet2DModelGrey, self).__init__()
        assert dims % 4 == 0
        
    def DownBlocks2D():
        pass

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
    import matplotlib.pyplot as plt
    test_se = SinusoidalEmbedding(512, 1000)
    print(test_se.embedding_table)
    plt.imshow(test_se.embedding_table, 'viridis')
    plt.axis('off') # Optional: hides axes
    plt.savefig('image.png')
