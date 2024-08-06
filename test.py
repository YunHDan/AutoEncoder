import matplotlib.pyplot as plt
from train import Autuencoder
from PIL import Image
import torch
import numpy as np

auto = torch.load('autoencoder.pkl')
auto = auto.cuda()
for i in range(12):
    input = np.array(Image.open(f'Data/img{i}.png'))
    input = torch.tensor(np.array([input]), dtype=torch.float32).permute(0, 3, 1, 2).cuda()
    out = auto(input)
    out = out.permute(0, 2, 3, 1)[0]
    out = out.detach().cpu().numpy()
    img = Image.fromarray(out.astype('uint8'))
    plt.imshow(img)
    plt.show()