import torch
import torch.nn.functional as F
image = torch.rand(6,6)
print("image=", image)
image = image.unsqueeze(dim=0)
print("image.shape=", image.shape)
image = image.unsqueeze(dim=0)
print("image.shape=", image.shape)
print("image=", image)
kernel = torch.ones(3,3)
print("kernel=", kernel)
kernel = kernel.unsqueeze(dim=0)
kernel = kernel.unsqueeze(dim=0)
kernel2=torch.randn(3,1,3,3,)
outimage2 = F.conv2d(image, kernel2,stride=1, padding=0)
print("outimage=", outimage2)
print("outimage.shape",outimage2.shape)
conv_layer=torch.nn.Conv2d(in_channels=1,out_channels=3,kernel_size=3,stride=1,padding=0)
out_image=conv_layer(image)
print("outimage=", out_image)
print("outimage.shape",out_image.shape)
parameters = list(conv_layer.parameters())
num_params = sum(p.numel() for p in parameters)

print(f"Number of parameters in the convolutional layer: {num_params}")