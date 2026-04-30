Hi yall, this is Lance

This is my repo for trying to train a diffusion model from the ground up

Enjoy ;P

--------------------

branch model/custom_mnist_model

I will write my own parameterised models on this branch, focus my study on the model side. 

The model I will imitate is the one from DDPM, it is a:

U-Net with sinusoidal timestep embedding and no classes

It will replace SimpleUNet2DModelGrey

--------------------

branch training/cifar10

I will use readily available diffusion models on this branch, focus my study on the training side. 

The model I use here is diffusers.UNet2DModel

Training with parameterized linear schedule, mnist, and cifar10 datasets are implemented here