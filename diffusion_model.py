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
        self.embedding_table = torch.zeros((max_T+1, embedding_dim))
        arr_i = torch.arange(embedding_dim//2)
        arr_1000_2i_d = torch.pow(10000, (2*arr_i / embedding_dim))
        for t in range(max_T+1):
            # the even indicies
            self.embedding_table[t, 0::2] = torch.sin(t/arr_1000_2i_d)
            self.embedding_table[t, 1::2] = torch.cos(t/arr_1000_2i_d)
        
        # print(self.embedding_dim)
        
    def forward(self, timesteps):
        assert torch.is_tensor(timesteps) and torch.all(timesteps <= self.max_T)
        # input: batched timestep of dimension [B], each t <= max_T
        # output: sinusoidal embeddings of dimension [B, self.embedding_dim]
        # B = timesteps.shape[0]
        # output = torch.zeros((B, self.embedding_dim))
        return self.embedding_table[timesteps]

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        assert out_channel % 32 == 0 and in_channel % 32 == 0
        super(ResBlock, self).__init__()
        self.group_norm1 = nn.GroupNorm(32, in_channel)
        self.non_linear1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 
                              kernel_size=3,
                              padding=1, 
                              bias=False)
        

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
    plt.imshow(test_se.embedding_table)
    plt.axis('off') # Optional: hides axes
    plt.savefig('image.png')
