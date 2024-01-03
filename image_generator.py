import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import random
from louis_cifar import DDPM 
import louis_cifar
import numpy as np
import torch

#define model parameters
n_T = 400 
device=torch.device("cpu")
n_classes = 10
n_feat = 128 
lrate = 1e-4
save_model = True

#load pretrained diffusion model 
ddpm = DDPM(nn_model=louis_cifar.ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=torch.device("cpu"), drop_prob=0.1)
ddpm.load_state_dict(torch.load("model_90.pth", map_location='cpu'))

#define desired generated image category/categories e.g. cat & plane 
context_label = [2,1]
c = torch.tensor([context_label], dtype=torch.long)

#generate image 
ddpm.eval()
with torch.no_grad():
    x_gen, x_gen_store = ddpm.sample_single(1, (3, 32, 32), device, c, guide_w=4.0)
    numpy_array = x_gen[0].squeeze(0).cpu().detach().numpy()
    numpy_array = np.transpose(numpy_array, (1, 2, 0))  # Transpose to (32, 32, 3)
    plt.imshow(numpy_array, cmap='gray')
    plt.axis('off')
    plt.savefig('output_image.png', bbox_inches='tight')
    plt.show()