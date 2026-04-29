from diffusers import UNet2DModel

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
